import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import random
import math
from scipy.spatial import ConvexHull
import time

class LocalizeBox():
    def __init__(self, d_img, camera_mat, appr_dist, debug=False):
        self.d_img = d_img
        self.camera_mat = camera_mat
        self.camera_mat[0][1] = 0.0
        self.camera_mat[1][0] = 0.0
        self.appr_dist = appr_dist
        self.debug = debug
    
    def pointcloud_from_depth(self, d_img, intr_mat):
        height, width = d_img.shape
        u, v = np.meshgrid(range(height), range(width), indexing='ij')
        z = d_img[u, v]
        mask = np.logical_and(z > 300, z < 800) # Apply mask to filter out unwanted points
        x = (u - intr_mat[0][2]) * z / intr_mat[0][0]
        y = (v - intr_mat[1][2]) * z / intr_mat[1][1]
        pointcloud = np.column_stack((x[mask], y[mask], z[mask]))
        return pointcloud

    def visualize_planes(self, planes, colors):
        # Plots open3d type planes 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(planes)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    def pcd_fit_plane(self, points, threshold, init_n, iter):
        # Fit plane with open3d segment_plane() function
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        w, index = pcd.segment_plane(threshold, init_n, iter)
        return w, index

    def color_plane(self, points, color):
        # Creates a color matrix for a set of points
        colors = np.full((len(points), 3), color)
        return colors

    def point_on_plane(self, coeffs, x, y):
        # Based on input x and y find the corresponding z value on a plane based on equation coefficients
        a, b, c, d = coeffs
        d = -d
        z = (d - a*x - b*y) / c
        return np.array([x,y,z])

    def plane_orientation(self, coeffs, deg=False):
        U = coeffs[0:3]
        U = -U # Flip the unit vector because we want to approach from the opposite direction
        # Find angles to camera axis of unit vector 
        ang_x = math.atan2(U[1], U[2]) # since unit vector we know that hypotenuse is always 1
        ang_y = math.atan2(U[0], U[2])
        ang_z = math.atan2(U[1], U[0])
        # Return either degrees or radians
        if not deg:
            return [ang_x, ang_y, ang_z]
        else:
            ang_x = self.rad_to_deg(ang_x)
            ang_y = self.rad_to_deg(ang_y)
            ang_z = self.rad_to_deg(ang_z)
            return [ang_x, ang_y, ang_z]

    def rad_to_deg(self, rad):
        # Converts a radian to a degree
        return rad*180/math.pi

    def find_planes(self, pcd, threshold=10, init_n=3, iterations=1000):
        # Algorithm for finding planes in a point cloud, iteratively fits a plane with ransac to a point cloud and removes inliers from the point cloud for next iteration
        planes = []
        plane_points = []
        colors = []
        pcd_np = np.asarray(pcd)
        available_points = pcd_np.copy()
        unsuccesful_count = 0 # Counter to see how many consecutive iterations it has failed at finding a sufficiently large plane
        while True:
            w, idx = self.pcd_fit_plane(available_points, threshold, init_n, iterations)
            color = self.color_plane(idx, [random.random(), random.random(), random.random()])
            
            # If the plane is fitted to more than 1000 points we accept it as an actual plane
            if len(idx) > 1000:
                planes.append(w)
                plane_points.append(available_points[idx])
                colors.append(color)
                available_points = np.delete(available_points, idx, axis=0)
                unsuccesful_count = 0
            else:
                unsuccesful_count += 1

            if unsuccesful_count > 5:
                break
        
        return planes, plane_points, colors



    def sort_planes_vertical_horizontal(self, planes, thresh=15):
        # Returns indices for vertical and horizontal planes based on a threshold
        vertical_id = []
        horizontal_id = []
        for i in range(len(planes)):
            _, y, _ = self.plane_orientation(planes[i], deg=True)
            if (y < 90+thresh and y > 90-thresh) or (y < -90+thresh and y > -90-thresh):
                horizontal_id.append(i)
            else:
                vertical_id.append(i)
        return vertical_id, horizontal_id

    def pcd_remove_outlier(self, points, neighbors, std):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, idx = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std)
        points = points[idx]
        return points

    def pcd_remove_outlier2(self, points, nb_points, radius):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        points = points[idx]
        return points

    def project_plane_to_2d(self, points, coeffs):
        normal = -coeffs[0:3]
        p_origin = self.plane_origin(points)
        normal = normal + p_origin
        # Find basis for 2D plane by creating two orthogonal vectors to normal
        b1 = np.array([-normal[1], normal[0], 0]) # The first orthogonal can be pretty much anywhere so we just rearrange the normal
        b2 = np.cross(normal, b1) # Taking the cross product of the two orthogonal vectors creates a third orthogonal to both
        N = np.transpose(np.array([b1,b2]))

        # Create projection matrix: P = A*(A'*A)^-1*A'
        Nt = np.transpose(N)
        NtN = np.matmul(Nt, N)
        inv_NtN = np.linalg.inv(NtN)
        P = np.matmul(np.matmul(N, inv_NtN), Nt)
        
        # Project points onto plane
        projected_points = np.matmul(P, np.array(points).T).T

        # Compute 2D points based on normal basis
        points_in_2d = np.matmul(N.T, projected_points.T).T

        return projected_points, points_in_2d



    def rot_mat_2d(self, angle):
        R = np.array([[math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]])
        return R


    def rot_mat(self, axis, angle):
        axes = ["X", "x", "Y", "y", "Z", "z"]
        assert axis in axes, "Invalid axis for rotation, choose either: ".format(axes)

        if axis in axes[0:2]: # If X rotation
            R = np.array([[1,               0,                0],
                        [0, math.cos(angle), -math.sin(angle)],
                        [0, math.sin(angle),  math.cos(angle)]])
            return R
        elif axis in axes[2:4]: # Else if Y rotation
            R = np.array([[ math.cos(angle), 0, math.sin(angle)],
                        [               0, 1,               0],
                        [-math.sin(angle), 0, math.cos(angle)]])
            return R
        else: # Else Z rotation
            R = np.array([[math.cos(angle), -math.sin(angle), 0],
                        [math.sin(angle),  math.cos(angle), 0],
                        [              0,                0, 1]])
            return R
        

    def euc_dist_2d(self, A, B):
        x1, y1 = A
        x2, y2 = B
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def plane_origin(self, data_points):
        avg_x = np.sum(data_points[:,0])/len(data_points[:,0])
        avg_y = np.sum(data_points[:,1])/len(data_points[:,1])
        avg_z = np.sum(data_points[:,2])/len(data_points[:,2])
        return np.array([avg_x, avg_y, avg_z])



    def main(self):
        start_t = time.time()
        # Create point cloud
        pcd = self.pointcloud_from_depth(self.d_img, self.camera_mat)
        
        # Find planes in point cloud
        planes, plane_points, colors = self.find_planes(pcd, 5, 3, 3000) # threshold 5 = 5mm +/-

        # Sort planes into approximate vertical or horizontal 
        v_idx, h_idx = self.sort_planes_vertical_horizontal(planes)
        if len(v_idx) > 2:
            #print("Caution: More than 2 vertical planes!")
            pass

        # Shelf face is the vertical plane closest to the camera
        dist_to_planes = []
        for idx in v_idx:
            # Check the z value at (x,y)=(0,0)
            P = self.point_on_plane(planes[idx], 0, 0)
            dist_to_planes.append(P[2])
        minimizer = np.argmin(P)
        min_idx = v_idx[minimizer]
        shelf_face = [planes[min_idx], plane_points[min_idx], colors[min_idx]]

        # Remove shelf face from list of vertical planes
        v_idx = np.delete(v_idx, minimizer)
        dist_to_planes = np.delete(dist_to_planes, minimizer)
        
        # Check if there is another vertical plane within X cm from the closest plane
        if len(v_idx) == 0: # If there are no more vertical planes we did not find a container at the location
            print("WARNING: No container found at shelf location!")
            return -1
        elif len(v_idx) == 1: # If there is only one other vertical plane we assume that is the container
            idx = v_idx[0]
            container_plane = [planes[idx], plane_points[idx], colors[idx]]
        else: # If there are multiple, we take the second closest plane - TODO: Add a check to make sure that the second closest is not a v2 of the shelf face
            minimizer = np.argmin(P)
            min_idx = v_idx[minimizer]
            container_plane = [planes[min_idx], plane_points[min_idx], colors[min_idx]]

        # Filter container plane to remove potential outliers (and their colors)
        len1 = len(container_plane[1])
        #container_plane[1] = self.pcd_remove_outlier(container_plane[1], 5, 2.0) # Remove outliers
        container_plane[1] = self.pcd_remove_outlier2(container_plane[1], 25, 10.0) # Remove outliers
        len2 = len(container_plane[1])
        container_plane[2] = container_plane[2][len1-len2:]
        #container_plane[2] = np.delete(container_plane[2], np.arange(0,len1-len2,1), axis=0) # Remove color of outliers

        # Project container points onto container plane
        container_projected, points_2d = self.project_plane_to_2d(container_plane[1], container_plane[0])
        #cp = self.color_plane(container_projected, [0,0,0])

        # Rotate points to create 2D representation where Z = 0 for all
        container_rot = self.plane_orientation(container_plane[0])
        Rx = self.rot_mat("X", -container_rot[0]) # Minus because we want to reverse their rotation
        Ry = self.rot_mat("Y", -container_rot[1])
        #Rz = self.rot_mat("Z", -container_rot[2])
        rot_container_points = np.matmul(Ry, np.matmul(Rx, np.array(container_projected).T)).T

        # Compute convex hull for 2D data
        reduced_dim_points = rot_container_points[:,0:2]
        hull = ConvexHull(reduced_dim_points)
        hull_points = reduced_dim_points[hull.vertices]

        # Compute bounding box for 2D data
        x_min = np.min(hull_points[:,0])
        y_min = np.min(hull_points[:,1])
        x_max = np.max(hull_points[:,0])
        y_max = np.max(hull_points[:,1])
        c1 = [x_min, y_min]
        c2 = [x_min, y_max]
        c3 = [x_max, y_min]
        c4 = [x_max, y_max]
        bbox_corners = [c1, c2, c3, c4]
        bbox_corners = np.asarray(bbox_corners)
        
        # Find closest hull vertices to bbox corners
        bbox_hull = np.zeros([4,2])
        i = 0
        for pa in bbox_corners:
            distances = np.sqrt(np.sum((pa - hull_points)**2, axis=1)) # Compute Euclidean distances from each corner of the bounding box to all the points in the convex hull
            min_candidate = hull_points[np.argmin(distances)] # Extract the minimizer from the hull points
            bbox_hull[i,:] = min_candidate # Append to array
            i += 1

        # Compute width of container based on bounding box
        topleft, topright = bbox_hull[2:4]
        #grasp_width = self.euc_dist_2d(topleft, topright)

        # Compute grasping point in 2D
        mid_point = [(topleft[0]+topright[0])/2, (topleft[1]+topright[1])/2]

        # Do the opposite rotation to the midpoint assuming Z = 0
        Rx = self.rot_mat("X", container_rot[0]) # Plus because we want to re-reverse their rotation
        Ry = self.rot_mat("Y", container_rot[1])
        mid_point = [mid_point[0], mid_point[1], 0] 
        mid_point_rot = np.matmul(Rx, mid_point)
        mid_point_rot = np.matmul(Ry, mid_point_rot)
        #mp_c = self.color_plane([mid_point_rot], [1, 0, 0])

        # Add plane origin to midpoint
        p_origin = self.plane_origin(container_plane[1])
        mid_point_o = p_origin+mid_point_rot # The point is given in mm relative to camera origin
        mid_point_o[1] = -mid_point_o[1]

        # Compute approach point by scaling negative plane normal and adding it to the target point
        # Unit normal is 1 mm, so scale by approach distance in mm
        appr_point = -container_plane[0][0:3]*(self.appr_dist*1000) + mid_point_o

        #print("Execution time:", time.time()-start_t)

        # FOR DEBUGGING
        debug = False
        if debug:
            plane_points = np.concatenate([shelf_face[1], 
                                    container_plane[1], 
                                    ], axis=0)
            colors = np.concatenate([shelf_face[2], 
                                    container_plane[2], 
                                    ], axis=0)
            
            self.visualize_planes(plane_points, colors)
        

        #print(mid_point_o)

        return mid_point_o, container_rot, appr_point

