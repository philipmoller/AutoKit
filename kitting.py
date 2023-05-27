from __future__ import print_function

import datetime
import argparse
import sys
import cv2
from cv2 import aruco
import numpy as np
import datetime
import time
from scipy import ndimage
import math
from matplotlib import pyplot as plt
import torch
import pandas as pd

from bosdyn.api import image_pb2
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2, manipulation_api_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, get_vision_tform_body, BODY_FRAME_NAME, HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client import math_helpers

import bosdyn.client.lease
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2, geometry_pb2, gripper_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.lease import LeaseClient
from bosdyn.util import seconds_to_duration

from localize_box import LocalizeBox

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf import wrappers_pb2

class Kitting():
    def __init__(self, argv):
        """
        Initialize Spot and grab control of lease clients
        """
        parser = argparse.ArgumentParser()
        bosdyn.client.util.add_base_arguments(parser)
        options = parser.parse_args(argv)

        bosdyn.client.util.setup_logging(options.verbose)

        self.sdk = bosdyn.client.create_standard_sdk('ArmTrajectory')
        self.robot = self.sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        assert not self.robot.is_estopped(), "Robot is estopped!"

        self.motion_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.world_object_client = self.robot.ensure_client(WorldObjectClient.default_service_name)

        """
        Class globabl variables
        """
        self.intr_mat = self.get_hand_intrinsics()

        self.aruco_dict = aruco.DICT_4X4_50
        self.aruco_shelf_size = 0.02
        self.aruco_grid_size = 0.005
        self.aruco_grid_ids = [11, 12, 13, 21, 22, 23, 31, 32, 33]
        self.aruco_ST_ids = [1, 2] # shelf and trolley IDs
        self.current_aruco_id = None

        self.appr_dist = 0.2

        self.gripper_search_height = 0.3

        self.height_offset = 0.0

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0

        # YOLOv5 model
        self.cnn_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.camera_bias = None 

        # Log paths
        self.log_folder_path = '/home/iemn/philip/Kitting/logs/'
        self.test_log = open(self.log_folder_path+"localization_test.txt", 'w')
        self.test_log = open(self.log_folder_path+"localization_test.txt", 'a')
        self.safety_log_path = '/home/iemn/philip/Kitting/logs/safety/'
        self.safety_log = open(self.safety_log_path+"safety_test.txt", 'w')
        self.safety_log = open(self.safety_log_path+"safety_test.txt", 'a')

        # Global orientation buffer
        self.global_rot = None

        # Global buffer for image source that most recently recognized a human
        self.priority_image_source = 'frontleft_fisheye_image'


    #####################################################################
    ####################### UTILITY FUNCTIONS ###########################
    #####################################################################
    def init_spot(self):
        """
        Turns Spot on and issues a blocking stand command
        """
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), "Robot power on failed."
        command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)

    def init_arm_left_height(self, height=0.3):
        """
        Un-stows the arm in front of Spot with a 45 degree rotation around the end-effector Z-axis and at a specified Z-axis translation.
        Used to place the arm in the rough direction of a shelf position marker
        """
        traj_time = 2
        position = [0.75, 0.0, height]
        rot = math_helpers.Quat.from_yaw(self.deg_to_rad(45))
        orientation = [rot.w, rot.x, rot.y, rot.z]
        self.movel(position, orientation, traj_time, True)


    def poweroff_spot(self):
        """
        Safe power-off option for Spot. Sits down before turning motors off
        """
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.robot.logger.info("Robot safely powered off.")

    def avg_point(self, points):
        """
        Returns the average 3D point from a list or array of 3D points
        """
        arr = np.asarray(points)
        avg_x = np.sum(arr[:,0])/len(arr[:,0])
        avg_y = np.sum(arr[:,1])/len(arr[:,1])
        avg_z = np.sum(arr[:,2])/len(arr[:,2])
        return np.array([avg_x, avg_y, avg_z])

    def deg_to_rad(self, deg):
        """
        Returns the radian value of a degree value
        """
        return deg*(math.pi/180)

    def quaternion_conjugate(self, q):
        """
        Returns the conjugate of a given quaternion by flipping the real axis
        """
        return np.array([q[0], -q[0], -q[0], -q[0]])
    
    def quaternion_length(self, q):
        """
        Returns the length of a given quaternion as the 2-norm
        """
        return q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2

    def quaternion_inverse(self, q):
        """
        Returns the inverse of a given quaternion
        """
        return self.quaternion_conjugate(q) / self.quaternion_length(q)

    def quaternion_angle_diff(self, q1, q2):
        """
        Returns the minimum arc angle between two quaternions, see:
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation
        """
        q1 = np.array([q1.w, q1.x, q1.y, q1.z])
        q2 = np.array([q2.w, q2.x, q2.y, q2.z])
        # find difference quaternion
        diff_quat = self.quaternion_inverse(q1)*q2
        # compute angle
        angle = 2 * math.atan2(math.sqrt(diff_quat[1]**2 + diff_quat[2]**2 + diff_quat[3]**2), diff_quat[0])
        return angle

    def pose_in_body_to_world(self, body_transform_object):
        """
        Returns an SE3Pose in the body frame to an SE3Pose in the worls frame
        """
        world_transform_body = self.get_body_tform_in_vision()
        world_transform_object = world_transform_body.mult(body_transform_object)
        return world_transform_object






    #####################################################################
    ###################### PERCEPTION COMMANDS ##########################
    #####################################################################
    def detect_aruco(self, source, size_of_marker, id):
        """
        Takes an image source as input, and from its image computes the location of an ID with a given size
        Returns the pose in the world frame as an SE3Pose
        """
        # Take the image
        image_request = [(build_image_request(source, quality_percent=100, pixel_format=None))]
        image_response = self.image_client.get_image(image_request)

        # Get camera matrix
        camera_mtx = self.get_camera_intrinsics(image_response[0])

        # Save the current transform snapshot
        transforms_snapshot = image_response[0].shot.transforms_snapshot

        # Decode the image
        image = np.frombuffer(image_response[0].shot.image.data, dtype=np.uint8)
        image = cv2.imdecode(image, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        aruco_dict = aruco.getPredefinedDictionary(self.aruco_dict)
        parameters =  aruco.DetectorParameters()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(image)

        try:
            if id not in ids:
                print("ID not found!")
                return -1, source
        except:
            print("ID not found!")
            return -1, source

        dist_coeffs = np.zeros((5,1))
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners=corners, markerLength=size_of_marker, cameraMatrix=camera_mtx, distCoeffs=dist_coeffs)
        rmats = []
        if rvecs is None:
            cv2.imshow('Failed ArUco reading', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            raise RuntimeError('Error: ArUco marker not found!')
        # Convert rotation vectors to matrices and draw the coordinate frame axis
        for i in range(len(tvecs)):
            rmat, _ = cv2.Rodrigues(rvecs[i])
            # We only use 1 ArUco marker, so save the right id
            if ids[i] == id:
                rmats = rmat
                tvecs = tvecs[i]
                self.current_aruco_id = id

        camera_tform_aruco = math_helpers.SE3Pose(x=tvecs[0][0]*10, y=tvecs[0][1]*10, z=tvecs[0][2]*10, rot=math_helpers.Quat.from_matrix(rmats))
        vision_tform_camera = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, 'hand_color_image_sensor')
        vision_tform_aruco = vision_tform_camera.mult(camera_tform_aruco)

        return vision_tform_aruco, source

    def detect_aruco_in_body(self, source, size_of_marker, id):
        """
        Same as above, but returns the pose in the body frame instead
        """
        # Take the image
        image_response = self.image_client.get_image_from_sources([source])[0]

        # Get the camera matrix information from the image response
        intrinsics = image_response.source.pinhole.intrinsics
        focal_length = intrinsics.focal_length
        principal_point = intrinsics.principal_point
        skew = intrinsics.skew
        camera_mtx = np.array([ [focal_length.x, skew.x, principal_point.x],
                                [skew.y, focal_length.y, principal_point.y],
                                [0, 0, 1]])

        # Decode the image
        dtype = np.uint8
        image = np.fromstring(image_response.shot.image.data, dtype=dtype)
        image = cv2.imdecode(image, -1)
        # Save the current transform snapshot
        transforms_snapshot = image_response.shot.transforms_snapshot

        aruco_dict = aruco.getPredefinedDictionary(self.aruco_dict)
        parameters =  aruco.DetectorParameters()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(image)

        #print("Visible ArUco markers:", ids)
        try:
            if id not in ids:
                print("ID not found!")
                return -1, source
        except:
            print("ID not found!")
            return -1, source

        # Estimate the transform between the camera and the ArUco markers 
        # TO DO: Change this to only estimate for the ID we are interested in 
        dist_coeffs = np.zeros((5,1))
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners=corners, markerLength=size_of_marker, cameraMatrix=camera_mtx, distCoeffs=dist_coeffs)
        rmats = []
        if rvecs is None:
            cv2.imshow('Failed ArUco reading', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            raise RuntimeError('Error: ArUco marker not found!')
        # Convert rotation vectors to matrices and draw the coordinate frame axis
        for i in range(len(tvecs)):
            rmat, _ = cv2.Rodrigues(rvecs[i])
            # We only use 1 ArUco marker, so save the right id
            if ids[i] == id:
                rmats = rmat
                tvecs = tvecs[i]
                self.current_aruco_id = id

        # Create camera-aruco transform from rotation matrix and translation vector (multiply values by 10 because this camera is configured with weird units)
        camera_tform_aruco = math_helpers.SE3Pose(x=tvecs[0][0]*10, y=tvecs[0][1]*10, z=tvecs[0][2]*10, rot=math_helpers.Quat.from_matrix(rmats))

        # Get the vision-camera transform
        body_tform_camera = get_a_tform_b(transforms_snapshot, BODY_FRAME_NAME, 'hand_color_image_sensor')
    
        # Combine the transforms to get vision-aruco transform
        body_tform_aruco = body_tform_camera.mult(camera_tform_aruco)

        return body_tform_aruco, source


    def pos_gripper_to_body(self, tf_snapshot, pos_vision, rot_vision=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)):
        """
        Returns a pose found in the camera from in the body frame
        """
        # Define vision pose as SE3Pose
        handcolor_tform_pos = math_helpers.SE3Pose(x=pos_vision[0], y=pos_vision[1], z=pos_vision[2], rot=rot_vision)

        # Get the vision-camera transform
        body_tform_handcolor = get_a_tform_b(tf_snapshot, BODY_FRAME_NAME, 'hand_depth_sensor')
    
        # Combine the transforms to get vision-aruco transform
        body_tform_aruco = body_tform_handcolor.mult(handcolor_tform_pos)
        return body_tform_aruco

    def get_box_location(self, debug=False):
        """
        Localizes the container on the shelf based on depth data from the hand camera
        """
        # Get gripper color and depth images
        color_image_request = [build_image_request('hand_color_in_hand_depth_frame', pixel_format=None)]
        depth_image_request = [build_image_request('hand_depth', pixel_format=None)]
        color_image_response = self.image_client.get_image(color_image_request)
        depth_image_response = self.image_client.get_image(depth_image_request)
        # Save the current transform snapshot
        transforms_snapshot = color_image_response[0].shot.transforms_snapshot

        # Get pixel byte lists and convert to images
        img = np.frombuffer(color_image_response[0].shot.image.data, dtype=np.uint8)
        img = cv2.imdecode(img, -1)
        d_img = np.frombuffer(depth_image_response[0].shot.image.data, dtype=np.uint16)
        d_img = np.reshape(d_img, img.shape[0:2])

        # Rotate images -90 degrees
        img = ndimage.rotate(img, -90)
        d_img = ndimage.rotate(d_img, -90)

        intr_mat = self.get_camera_intrinsics(color_image_response[0])

        LB = LocalizeBox(d_img, intr_mat, self.appr_dist, debug)
        pos_list = []
        rot_list = []
        appr_list = []
        limit = 3
        for i in range(0,limit):
            pos, rot, appr = LB.main()
            pos_list.append(pos)
            rot_list.append(rot)
            appr_list.append(appr)
        pos = self.avg_point(pos_list)
        rot = self.avg_point(rot_list)
        appr = self.avg_point(appr_list)

        pos_in_body = self.pos_gripper_to_body(transforms_snapshot, pos/1000.0)
        appr_in_body = self.pos_gripper_to_body(transforms_snapshot, appr/1000.0)
        return pos_in_body, rot, appr_in_body

    def get_hand_intrinsics(self):
        """
        Returns the intrinsic matrix for the hand camera
        """
        color_image_request = [build_image_request('hand_color_in_hand_depth_frame', pixel_format=None)]
        color_image_response = self.image_client.get_image(color_image_request)
        fx = color_image_response[0].source.pinhole.intrinsics.focal_length.x
        fx = color_image_response[0].source.pinhole.intrinsics.focal_length.x
        fy = color_image_response[0].source.pinhole.intrinsics.focal_length.y
        cx = color_image_response[0].source.pinhole.intrinsics.principal_point.x
        cy = color_image_response[0].source.pinhole.intrinsics.principal_point.y
        skew_x = color_image_response[0].source.pinhole.intrinsics.skew.x
        skew_y = color_image_response[0].source.pinhole.intrinsics.skew.y
        intr_mat = np.array([[fx, skew_x, cx],
                            [skew_y, fy, cy],
                            [0,  0,  1]])
        return intr_mat

    def get_camera_intrinsics(self, image_response):
        """
        Returns the intrinsic matrix for a given image response
        """
        ints = image_response.source.pinhole.intrinsics
        intr_mat = np.array([[ints.focal_length.x, ints.skew.x,         ints.principal_point.x],
                             [ints.skew.y,         ints.focal_length.y, ints.principal_point.y],
                             [0,                   0,                   1]])
        return intr_mat

    def locate_aruco_from_body(self, id, size_of_marker):
        """
        Scans through all 5 body cameras and returns the position of a given marker ID if found
        """
        # List of body camera references
        image_sources = ['frontleft_fisheye_image',
                         'frontright_fisheye_image',
                         'left_fisheye_image',
                         'right_fisheye_image',
                         'back_fisheye_image']

        tf_sources = ['frontleft_fisheye',
                      'frontright_fisheye',
                      'left_fisheye',
                      'right_fisheye',
                      'back_fisheye']

        # Grab images and their transforms
        images = []
        intrinsics = []
        transforms = []
        for source in image_sources:
            image_request = [(build_image_request(source, quality_percent=100, pixel_format=None))]
            image_response = self.image_client.get_image(image_request)
            image = np.frombuffer(image_response[0].shot.image.data, dtype=np.uint8)
            images.append(cv2.imdecode(image, -1))
            intrinsics.append(self.get_camera_intrinsics(image_response[0]))
            transforms.append(image_response[0].shot.transforms_snapshot)
                
        aruco_dict = aruco.getPredefinedDictionary(self.aruco_dict)
        parameters =  aruco.DetectorParameters()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        id_located = False
        for i in range(len(images)):
            #corners, ids, _ = aruco.detectMarkers(images[i], aruco_dict, parameters=parameters)
            corners, ids, _ = detector.detectMarkers(images[i])
            try:
                if id in ids: # If ID is present, compute location
                    id_located = True
                    dist_coeffs = np.zeros((5,1))
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners=corners, markerLength=size_of_marker, cameraMatrix=intrinsics[i], distCoeffs=dist_coeffs)
                    rmats = []
                    if rvecs is None:
                        cv2.imshow('Failed ArUco reading', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        raise RuntimeError('Error: ArUco marker not found!')
                    # Convert rotation vectors to matrices and draw the coordinate frame axis
                    for j in range(len(tvecs)):
                        rmat, _ = cv2.Rodrigues(rvecs[j])
                        # We only use 1 ArUco marker, so save the right id
                        if ids[j] == id:
                            rmats = rmat
                            tvecs = tvecs[j]
                    # Convert ArUco pose to SE3Pose
                    camera_tform_aruco = math_helpers.SE3Pose(x=tvecs[0][0]*10, y=tvecs[0][1]*10, z=tvecs[0][2]*10, rot=math_helpers.Quat.from_matrix(rmats))
                    # Get transformation from vision frame to camera frame
                    vision_tform_camera = get_a_tform_b(transforms[i], VISION_FRAME_NAME, tf_sources[i])
                    # Get transformation from vision frame to aruco marker
                    vision_tform_aruco = vision_tform_camera.mult(camera_tform_aruco)

                    return vision_tform_aruco, image_sources[i]
                else:
                    pass
            except Exception as e:
                print("Error:", e)
        if not id_located:
            timestamp = datetime.datetime.now()
            for i in range(len(images)):
                image_name = "LocalizationFail_{}_{}.png".format(tf_sources[i], timestamp)
                cv2.imwrite(self.log_folder_path+image_name, images[i])
        return -1

    def sort_image_sources(self, image_sources, priority_source):
        """
        Sorts a list of image sources and returns a list such that a priority source becomes first
        """
        priority_idx = image_sources.index(priority_source)
        buffed_source = image_sources[0]
        image_sources[0] = priority_source
        image_sources[priority_idx] = buffed_source
        return image_sources

    def check_for_human(self):
        """
        Safety layer to check for humans in the vicinity. Used to stop movement if a hazard is detected.
        """
        image_sources = ['frontleft_fisheye_image',
                         'frontright_fisheye_image',
                         'left_fisheye_image',
                         'right_fisheye_image',
                         'back_fisheye_image']

        image_sources = self.sort_image_sources(image_sources, self.priority_image_source)

        tf_sources = ['frontleft_fisheye',
                      'frontright_fisheye',
                      'left_fisheye',
                      'right_fisheye',
                      'back_fisheye']      

        depth_in_visual_sources = ['back_depth_in_visual_frame',
	                               'frontleft_depth_in_visual_frame',
                                   'frontright_depth_in_visual_frame',
                                   'left_depth_in_visual_frame',
                                   'right_depth_in_visual_frame']

        i = 0
        # Get images from all body cameras
        for source in image_sources:
            # Get image data
            image_request = [(build_image_request(source, quality_percent=100, pixel_format=None))]
            d_image_request = [(build_image_request(depth_in_visual_sources[i], quality_percent=100, pixel_format=None))]
            image_response = self.image_client.get_image(image_request)
            d_image_response = self.image_client.get_image(d_image_request)

            # Convert to images
            image = np.frombuffer(image_response[0].shot.image.data, dtype=np.uint8)
            image = cv2.imdecode(image, -1)
            d_image = np.frombuffer(d_image_response[0].shot.image.data, dtype=np.uint16)
            d_image = np.reshape(d_image, image.shape[0:2])
            # Approximately rotate to image
            if image_response[0].source.name[0:5] == 'front':
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif image_response[0].source.name[0:5] == 'right':
                image = cv2.rotate(image, cv2.ROTATE_180)
            # Do inference
            cnn_output = self.cnn_model(image)
            # Convert output to pandas dataframe
            pd_output = cnn_output.pandas().xyxy[0]
            # Extract present classes and check if there are any humans
            classes = pd_output['class'].values
            humans = np.where(classes==0)[0] # person label = 0
            # If there are no humans present in the image, pass and continue to the next image source
            if len(humans) == 0:
                if image_response[0].source.name[0:5] == 'front':
                    cv2.imwrite(self.safety_log_path+source+"{}.png".format(datetime.datetime.now()), image)
            # Else, check if the human is closer than 2m
            else:
                # For debugging
                return_image = np.squeeze(cnn_output.render())
                intr_mat = self.get_camera_intrinsics(image_response[0])
                transforms_snapshot = image_response[0].shot.transforms_snapshot
                # For each human present, check the confidence level to make sure that it is a human
                for human in humans:
                    conf = pd_output['confidence'].values[human]
                    # If significantly high confidence, estimate the center
                    if conf > 0.25:
                        xmin = int(pd_output['xmin'].values[human])
                        xmax = int(pd_output['xmax'].values[human])
                        ymin = int(pd_output['ymin'].values[human])
                        ymax = int(pd_output['ymax'].values[human])

                        # TODO: check that the return from min_depth is not a 0 i.e. a bad value
                        d_bbox = d_image[ymin:ymax][xmin:xmax]
                        good_idx = np.where(d_bbox > 0.0)
                        good_values = d_bbox[good_idx]
                        try:
                            min_depth_in_bbox = np.min(good_values)/1000.0 # minimum depth pixel in bbox converted to meter

                            xmid = int(round((xmin+xmax)/2.0))
                            ymid = int(round((ymin+ymax)/2.0))

                            z = min_depth_in_bbox
                            x = (xmid - intr_mat[0][2]) * z / intr_mat[0][0]
                            y = (ymid - intr_mat[1][2]) * z / intr_mat[1][1]
                            camera_to_point = math_helpers.SE3Pose(x=x, y=y, z=z, rot=math_helpers.Quat(w=0, x=0, y=0, z=0))
                            body_to_camera = get_a_tform_b(transforms_snapshot, BODY_FRAME_NAME, tf_sources[i])
                            body_to_point = body_to_camera.mult(camera_to_point)

                            world_to_body = self.get_body_tform_in_vision()
                            world_to_point = world_to_body.mult(body_to_point)
                            p3d = np.array([world_to_point.x, world_to_point.y, world_to_point.z])
                            b3d = np.array([world_to_body.x, world_to_body.y, world_to_body.z])

                            print("MIN DEPTH:", min_depth_in_bbox)
                            dist3d = self.euc_dist_3d(b3d, p3d)
                            min_depth_in_bbox = dist3d

                            print("MIN DISTANCE:", min_depth_in_bbox)

                            if min_depth_in_bbox < 1.5:
                                self.priority_image_source = source
                                return min_depth_in_bbox, return_image
                        except Exception as e:
                            print("Error in finding a minimum distance for human:")
                            print(e)
                            return 0, return_image

                        
                        #xmid = int(round((xmin+xmax)/2.0))
                        #ymid = int(round((ymin+ymax)/2.0))
                        #center = np.array([xmid, ymid])
                        #cv2.circle(return_image, center, 15, [255, 255, 255], -1)

                #cv2.imshow("{}".format(source), return_image)
                #cv2.imshow("{}".format(source), d_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            i += 1
        return 0, np.zeros(image.shape)



        


    #####################################################################
    ################## MANIPULATOR CONTROL COMMANDS #####################
    #####################################################################
    def movel(self, pos, q, t, blocking=False):
        """
        Move end-effector linearly to a given Cartesian and quaternion position. Optional bool to force the trajectory to block the robot until complete.
        """
        self.robot.time_sync.wait_for_sync()

        target_ori = geometry_pb2.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        target_pose = math_helpers.SE3Pose(x=pos[0], y=pos[1], z=pos[2], rot=target_ori)

        target_traj_point = trajectory_pb2.SE3TrajectoryPoint(pose=target_pose.to_proto(), time_since_reference=seconds_to_duration(t))
        target_traj = trajectory_pb2.SE3Trajectory(points=[target_traj_point])

        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(pose_trajectory_in_task=target_traj, root_frame_name=BODY_FRAME_NAME)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=arm_cartesian_command)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)
        cmd_id = self.command_client.robot_command(robot_command) # Execute command

        # If blocking is enabled this will force the robot to wait until trajectory is complete
        if blocking == True: # TO DO: Change to look at the command id instead
            start_t = time.time()
            while time.time()-start_t < t:
                pass
                

    def movej(self, q, t=5, deg=True):
        """
        Rotates the joints with the specified angles (in radians). The trajectory is executed in the specified time in seconds, defaults to 5 seconds.
        """
        assert len(q)==6, "Invalid joint input!"
        if deg == True:
            radians = []
            for angle in q:
                radians.append(self.deg_to_rad(angle))
            q1, q2, q3, q4, q5, q6 = radians
        else:
            q1, q2, q3, q4, q5, q6 = q

        if type(q1) == wrappers_pb2.DoubleValue:
            arm_position = arm_command_pb2.ArmJointPosition(sh0=q1, 
                                                            sh1=q2,
                                                            el0=q3, 
                                                            el1=q4,
                                                            wr0=q5, 
                                                            wr1=q6)
        else:            
            arm_position = arm_command_pb2.ArmJointPosition(sh0=wrappers_pb2.DoubleValue(value=q1), 
                                                            sh1=wrappers_pb2.DoubleValue(value=q2),
                                                            el0=wrappers_pb2.DoubleValue(value=q3), 
                                                            el1=wrappers_pb2.DoubleValue(value=q4),
                                                            wr0=wrappers_pb2.DoubleValue(value=q5), 
                                                            wr1=wrappers_pb2.DoubleValue(value=q6))
        # Wrap position in trajectory structure
        arm_joint_trajectory_point = arm_command_pb2.ArmJointTrajectoryPoint(position=arm_position,
                                                                             time_since_reference=seconds_to_duration(t))
        arm_joint_trajectory = arm_command_pb2.ArmJointTrajectory(points=[arm_joint_trajectory_point], 
                                                                  maximum_velocity=wrappers_pb2.DoubleValue(value=4.0))
        arm_joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_trajectory)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=arm_joint_move_command)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)
        # Issue command to robot
        cmd_id = self.command_client.robot_command(robot_command)

    def open_gripper(self,open_fraction=1.0):
        theta = open_fraction * (-1.5708) # Max opening angle in radians
        claw_gripper_command = gripper_command_pb2.ClawGripperCommand.Request(trajectory=trajectory_pb2.ScalarTrajectory(points=[trajectory_pb2.ScalarTrajectoryPoint(point=theta, time_since_reference=seconds_to_duration(1.0))]),
                                                                              maximum_open_close_velocity=wrappers_pb2.DoubleValue(value=6.28),
                                                                              maximum_torque=wrappers_pb2.DoubleValue(value=0.5))
        gripper_command = gripper_command_pb2.GripperCommand.Request(claw_gripper_command=claw_gripper_command)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(gripper_command=gripper_command)
        command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)
        cmd_id = self.command_client.robot_command(command)
        #t = time.time()
        #while time.time()-t < 2.0:
        #    time.sleep(0.05)
        #    print(self.command_client.robot_command_feedback(cmd_id))

    def arm_carry_position(self):
        """
        Places arm in pre-defined carry position for containers
        """
        traj_time = 2
        position = [0.65, 0.0, 0.35]
        orientation = [0.924, 0.0, -0.383, 0.0]
        self.movel(position, orientation, traj_time, True)

    def lock_arm_in_body(self):
        """
        Issues a joint command to the current joint configuration such that the end-effector is not locked at a world pose
        """
        joint_states = self.robot_state_client.get_robot_state().kinematic_state.joint_states
        arm_idx = [12, 13, 15, 16, 17, 18]
        joint_positions = []
        for idx in arm_idx:
            joint_positions.append(joint_states[idx].position)
        self.movej(joint_positions, t=1, deg=False)
        time.sleep(2)

    def tuck_arm(self):
        """
        Stow arm
        """
        stow = RobotCommandBuilder.arm_stow_command()
        self.command_client.robot_command(stow)
        time.sleep(2)

    def get_arm_joint_positions(self):
        """
        Read current joint configuration of arm from kinematic state
        """
        joint_states = self.robot_state_client.get_robot_state().kinematic_state.joint_states
        arm_idx = [12, 13, 15, 16, 17, 18]
        joint_positions = np.zeros(len(arm_idx))
        for i in range(len(arm_idx)):
            joint_positions[i] = joint_states[arm_idx[i]].position.value
        return joint_positions


    def get_gripper_position(self):
        open_percentage = self.robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage
        return open_percentage

    def force_side_align(self):
        Q = self.get_arm_joint_positions()
        Q[3] = Q[3] + math.pi/2.0
        Q[4] = Q[4] + math.pi/4.0
        Q[5] = Q[5] - math.pi/2.0
        self.movej(Q, t=2, deg=False)
        time.sleep(3)

    #####################################################################
    ####################### MOBILITY COMMANDS ###########################
    #####################################################################
    def move_simple(self, pose, end_time):
        """
        Moves to a SE3Pose with a given maximum time
        """
        target_x = pose.x
        target_y = pose.y
        heading = pose.rot.to_yaw()
        # Command the robot to go to the tag in kinematic odometry frame
        speed_limit = geometry_pb2.SE2VelocityLimit(max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
        mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit, locomotion_hint=spot_command_pb2.HINT_AUTO)
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=target_x, 
                                                                           goal_y=target_y,
                                                                           goal_heading=heading, 
                                                                           frame_name=VISION_FRAME_NAME, 
                                                                           params=mobility_params,
                                                                           body_height=0.0, 
                                                                           locomotion_hint=spot_command_pb2.HINT_AUTO)
        # Issue the command to the robot
        self.command_client.robot_command(lease=None, 
                                          command=tag_cmd,
                                          end_time_secs=time.time() + end_time)
        time.sleep(end_time)
        return

    def move_to_NoWait(self, pose, dist_offset=0.5, side_offset=0.0, angle=0.0):
        """
        Function to move to vertically placed ArUco markers with a specific offset and angle

        Input
            pose: SE3Pose, defining the pose of a an object in the world frame
            offset: double, offset along z axis for approach point
            angle: double, body alignment to goal pose given in degrees
            thresholds: [double, double], precision thresholds for when to accept the movement as done, first entry is position, second is rotation
        """

        # Create an approach point by offsetting the goal point along the Z axis
        R = pose.rot.to_matrix()
        z_axis = R[:,2] # THIS IS 1M UNIT VECTOR
        x_axis = R[:,0] 
        pose_arr = np.array([pose.x, pose.y, pose.z])
        target = pose_arr + (dist_offset*z_axis) + (side_offset*x_axis)
        target_x = target[0]
        target_y = target[1]

        # Compute heading from approach pose to marker pose and add desired angle
        approach_to_marker = pose_arr[0:2] - target[0:2]
        angle_in_frame = math.atan2(approach_to_marker[1], approach_to_marker[0])
        heading = angle_in_frame + (angle*(math.pi/180))

        # Command the robot to go to the tag 
        speed_limit = geometry_pb2.SE2VelocityLimit(max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
        mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit, locomotion_hint=spot_command_pb2.HINT_AUTO)
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=target_x, 
                                                                           goal_y=target_y,
                                                                           goal_heading=heading, 
                                                                           frame_name=VISION_FRAME_NAME, 
                                                                           params=mobility_params,
                                                                           body_height=0.0, 
                                                                           locomotion_hint=spot_command_pb2.HINT_AUTO)
        end_time = 10.0 # this doesnt matter since we pause in the while loop
        # Issue the command to the robot
        self.command_client.robot_command(lease=None, 
                                          command=tag_cmd,
                                          end_time_secs=time.time() + end_time)
        return target, heading

    def turn_body_45(self):
        """
        Function for turning the body 45 degrees to where it currently is focused around keeping the head of Spot in the same place
        """
        # Extract current pose
        original_pose = self.get_body_tform_in_vision()
        original_rot = original_pose.rot.to_matrix()
        original_yaw = original_pose.rot.to_yaw()
        # Extract axis vectors
        x_axis = original_rot[:,0]
        y_axis = original_rot[:,1]
        
        new_yaw = original_yaw + (-45*(math.pi/180))
        new_rot = math_helpers.Quat.from_yaw(new_yaw)
        new_pos = np.array([original_pose.x, original_pose.y, original_pose.z]) + x_axis*0.4 + y_axis*0.5
        new_pose = math_helpers.SE3Pose(x=new_pos[0], y=new_pos[1], z=new_pos[2], rot=new_rot)
        self.move_simple(new_pose, 2)

    def rotate_body(self, angle, end_time=2):
        """
        Function for turning the body in place
        """
        # Extract current pose
        original_pose = self.get_body_tform_in_vision()
        original_yaw = original_pose.rot.to_yaw()
        
        new_yaw = original_yaw + (angle*(math.pi/180))
        new_rot = math_helpers.Quat.from_yaw(new_yaw)
        new_pose = math_helpers.SE3Pose(position=original_pose.position, rot=new_rot)
        self.move_simple(new_pose, end_time=2)

    def euc_dist_2d(self, A, B):
        x1, y1 = A
        x2, y2 = B
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def euc_dist_3d(self, A, B):
        x1, y1, z1 = A
        x2, y2, z2 = B
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def get_body_tform_in_vision(self):
        body_tform_in_vision = get_vision_tform_body(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
        return body_tform_in_vision



    #####################################################################
    ######################### ROBOT SKILLS ##############################
    #####################################################################
    def localize_ID(self, ID):
        """
        Localizes a shelf ArUco marker in Spot world frame.
        Returns 1 if found, returns 0 if not
        """
        try:
            aruco = self.locate_aruco_from_body(ID, self.aruco_shelf_size)
            if aruco == -1:
                print("Searching for ID {}...".format(ID))
                self.rotate_body(-30)
                aruco = self.locate_aruco_from_body(ID, self.aruco_shelf_size)
                if aruco == -1:
                    print("Could not locate ID {}!".format(ID))
                    return 0, None
            pose, source = aruco
            return 1, pose
        except Exception as e:
            print("Error:", e)
            print("Failed in localizing the ID!")
            return 0, None

    def move_to_aruco(self, ID, marker_size, dist_offset=1.0, side_offset=0.0, image_source='body', global_orientation=False, update_global=False):
        """
        Makes Spot move to a shelf head-on with a given offset. Spot looks at a new frame every few seconds to improve its estimates of the ArUco marker if it start out far away.
        Returns 1 if successful, returns 0 if failed.
        WARNING: This function may not work properly at offsets larger than 1.0m since ArUco estimates become too noisy
        """
        #try:
        # Find initial estimate of ArUco marker
        if image_source == 'body':
            aruco = self.locate_aruco_from_body(ID, marker_size)
        elif image_source == 'hand_color_image':
            aruco = self.detect_aruco(image_source, marker_size, ID)
        else:
            print("Invalid image source for ArUco detection! Accepted inputs are: 'body' or 'hand_color_image'")
            return 0
        
        if aruco[0] == -1:
            if image_source == 'hand_color_image':
                # If the ArUco marker is not found, search for it by turning left and right in place
                aruco = self.search_aruco_cone(ID, marker_size, image_source)
                if aruco == -1:
                    print("Could not locate ID {}!".format(ID))
                    return 0
            else:
                print("Failed to find ID: {}".format(ID))
                return 0
        if aruco[0] != -1:
            # Else, command the robot to go to target
            target, source = aruco
            
            if update_global:
                self.global_rot = target.rot

            if global_orientation:
                if self.global_rot == None:
                    print("No prior global orientation stored in buffer!")
                else:    
                    target.rot = self.global_rot

            target_pos, target_angle = self.move_to_NoWait(target, dist_offset, side_offset)

            moving = True
            start_t = time.time()
            while moving:
                """
                # Check if seconds have passed
                if time.time()-t > 3.0:
                    # If true, update t and look at new frame to estimate ArUco marker again
                    t = time.time()
                    aruco = self.locate_aruco_from_body(ID, self.aruco_shelf_size)
                    if aruco == -1:
                        # If the ArUco marker is not found, pass and continue with current target
                        pass
                    else:
                        # Else, command Spot to go to the updated target
                        target, source = aruco
                        target_pos, target_angle = self.move_to_NoWait(target, offset)
                else:
                """
                current_pos = self.get_body_tform_in_vision()
                current_angle = current_pos.rot.to_yaw()
                x_diff = abs(target_pos[0] - current_pos.x)
                y_diff = abs(target_pos[1] - current_pos.y)
                rot_diff = abs(target_angle - current_angle)
                if x_diff < 0.075 and y_diff < 0.075 and (rot_diff < 0.075 or rot_diff > (math.pi*2)-0.075):
                    moving = False
                    break
                elif time.time()-start_t > 7:
                    break

            self.gripper_search_height = target.z - current_pos.z
            return 1


    def move_to_trolley(self, ID, target_offset):
        """
        Makes Spot move to a pose with a given offset + 0.3m head-on, find the ID again and move to the target offset head-on.
        Returns 1 if successful, returns 0 if failed.
        """
        try:
            self.move_to_aruco(ID, marker_size=self.aruco_shelf_size, dist_offset=target_offset+0.3)
            time.sleep(0.25)
            self.move_to_aruco(ID, marker_size=self.aruco_shelf_size, dist_offset=target_offset+0.3)
            time.sleep(0.25)
            self.move_to_aruco(ID, marker_size=self.aruco_shelf_size, dist_offset=target_offset+0.3, side_offset=0.5)
            time.sleep(0.25)
            self.move_to_aruco(ID, marker_size=self.aruco_shelf_size, dist_offset=target_offset)
            time.sleep(0.25)
            return 1
        except Exception as e:
            print("Problem when moving to trolley:")
            print("Error:", e)
            return 0

    def align_to_shelf_location(self, ID):
        """
        This skill will find the shelf location for the desired container.
        If the container is not immediately visible, the robot will search for it and align its body to it.
        Returns 1 if the robot is succesfully aligned to it, returns 0 if the shelf location cannot be found.
        """
        try:
            # Initialize arm and move to home position
            #self.init_arm()
            self.init_arm_left_height(self.gripper_search_height)

            time.sleep(2.5)

            # Find aruco marker
            aruco_pos_in_body, source = self.detect_aruco_in_body('hand_color_image', self.aruco_grid_size, ID)

            top_row = np.array([11, 21, 31])
            if aruco_pos_in_body == -1 and ID in top_row:
                self.force_side_align()
                aruco_pos_in_body, source = self.detect_aruco_in_body('hand_color_image', self.aruco_grid_size, ID)

            if aruco_pos_in_body == -1:
                print("Failed to find ArUco marker with ID: {}".format(ID))
                return 0
            else:
                # Approach container
                rot = math_helpers.Quat.from_yaw(-aruco_pos_in_body.rot.to_yaw())
                rot = [rot.w, rot.x, rot.y, rot.z]
                approach_x_offset = -0.2
                approach_y_offset = -0.2
                pos = aruco_pos_in_body.position
                position = [pos.x+approach_x_offset, pos.y+approach_y_offset, pos.z]
                self.movel(position, rot, 2, True)
                return 1

        except Exception as e:
            print("Failed at aligning gripper with shelf location {}!".format(ID))
            print("Error:", e)
            return 0


    def localize_container(self):
        """
        Estimates the pose of the container on the shelf based on depth data.
        Returns 1 if successful, returns 0 if not.
        """
        try:
            left_angle = 45
            c_pos, c_rot, c_appr = self.get_box_location()
            c_rot = math_helpers.Quat.from_yaw(self.deg_to_rad(left_angle-c_rot[1]))
            c_rot = [c_rot.w, c_rot.x, c_rot.y, c_rot.z]
            return 1, c_pos, c_rot, c_appr
        except Exception as e:
            print("Failed at locating container!")
            print("Error:", e)
            return 0, None, None, None

    def grasp_container(self, target, rotation, approach):
        """
        Grasps a container with the arm, places it in a carry position and locks the arm joints.
        Returns 1 if successful, returns 0 if not.
        """
        try:
            # Values for aligning to TCP
            gripper_z = 0.015
            gripper_x = 0.025

            # Approach
            appr_pos = [approach.x, approach.y, approach.z+gripper_z] # ADD OFFSET ON Z VALUE FOR GRIPPER
            #appr_pos[1] = -appr_pos[1]
            self.movel(appr_pos, rotation, 2, True)
            
            # Target
            target = [target.x+gripper_x, target.y, target.z+gripper_z]
            #c_pos[1] = -c_pos[1]
            self.movel(target, rotation, 2, True)
            time.sleep(0.25)

            # Close gripper
            self.open_gripper(1.0)

            # Lift the container up a bit
            lift_dist = 0.02
            lift_pos = target.copy()
            lift_pos[2] += lift_dist
            self.movel(lift_pos, rotation, 2, True)

            # Pull it out
            retr_pos = appr_pos.copy()
            retr_pos[2] += lift_dist
            retr_pos[1] -= 0.15
            retr_pos[0] -= 0.15
            self.movel(retr_pos, rotation, 2, True)

            # Flip the wrist up to prevent dropping while moving
            self.arm_carry_position()

            # Lock joints
            self.lock_arm_in_body()
        
            gripper_pos = self.get_gripper_position()

            if gripper_pos > 95.0: # Grasp failed if the gripper completely closed 
                return -1
            else:
                return 1

        except Exception as e:
            print("Failed at grasping container!")
            print("Error:", e)
            return 0

    def place_on_trolley(self, i):
        """
        Places a container on the trolley at a hard-coded position.
        Returns 1 if successful, returns 0 if not.
        """
        height_offset = 0.0
        poses = [[[0.85,  0.275, 0.47+height_offset], [0.85,  0.275, 0.37+height_offset], [0.65,  0.275, 0.37+height_offset]],
                 [[0.85,  0.275, 0.62+height_offset], [0.85,  0.275, 0.47+height_offset], [0.65,  0.275, 0.47+height_offset]],
                 [[0.85, -0.225, 0.47+height_offset], [0.85, -0.225, 0.37+height_offset], [0.65, -0.225, 0.37+height_offset]],
                 [[0.85, -0.225, 0.62+height_offset], [0.85, -0.225, 0.47+height_offset], [0.65, -0.225, 0.47+height_offset]],
                 [[0.85,  0.0,   0.47+height_offset], [0.85,  0.0,   0.37+height_offset], [0.65,  0.0,   0.37+height_offset]],
                 [[0.85,  0.0,   0.62+height_offset], [0.85,  0.0,   0.47+height_offset], [0.65,  0.0,   0.47+height_offset]]]

        current_poses = poses[i]
        orientation = [1.0, 0.0, 0.0, 0.0]
        try:
            # Approach
            self.movel(current_poses[0], orientation, 5, True)
            # Target
            self.movel(current_poses[1], orientation, 2, True)
            # Open gripper
            self.open_gripper(0.0)
            time.sleep(1)
            # Retract
            self.movel(current_poses[2], orientation, 2, True)
            time.sleep(0.5)          
            return 1
        
        except Exception as e:
            print("Error:", e)
            print("ERROR: Failed to place container on trolley!")
            return 0

    def search_aruco_cone(self, ID, marker_size, image_source):
        """
        Searches for an ArUco marker in a cone in front of Spot by turning left and right in place and getting a new image for each turn
        Returns 1 and the marker if found, returns 0 if not
        """
        print("Searching for ArUco ID {}...".format(ID))
        turn_angle = 15*(math.pi/180)
        # Get current pose
        original_pose = self.get_body_tform_in_vision()
        original_rot = original_pose.rot.to_yaw()
        # Compute left pose
        left_rot = original_rot + turn_angle
        left_rot = math_helpers.Quat.from_yaw(left_rot)
        left_pose = math_helpers.SE3Pose(x=original_pose.x, y=original_pose.y, z=original_pose.z, rot=left_rot)
        # Compute right pose
        right_rot = original_rot - turn_angle
        right_rot = math_helpers.Quat.from_yaw(right_rot)
        right_pose = math_helpers.SE3Pose(x=original_pose.x, y=original_pose.y, z=original_pose.z, rot=right_rot)
        # Command the robot to move between the poses and check for the ArUco ID
        poses = [left_pose, right_pose]
        for pose in poses:
            self.move_simple(pose, 2)
            aruco = self.detect_aruco(image_source, marker_size, ID)
            if aruco == -1:
                pass
            else:
                return aruco
        return -1



    #####################################################################
    ############################# TASKS #################################
    #####################################################################
    def kitting_task(self):
        shelf = 1
        trolley = 2

        #print("Enter 1 to start:")
        #start_input = int(input())
        #if start_input == 1:
        #    pass
        #else:
        #    return

        #containers = []
        #print("Input container 1:")
        #containers.append(int(input()))

        #print("Input container 2:")
        #containers.append(int(input()))
        containers = [12, 22, 32, 11, 21, 31]
        #containers = [22]
        
        debug = False
        if not debug:
            i = 0
            for container in containers:
                container_grasped = False
                grasp_attempts = 0
                align_attempts = 0
                while not container_grasped:
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return

                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return
                
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body', update_global=True)
                    if retval == 0:
                        print("Stopping program...")
                        return
                    
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, 1.25, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return

                    print("Approaching container")
                    # Approach shelf location
                    time.sleep(0.5)
                    retval = self.move_to_aruco(container, self.aruco_grid_size, dist_offset=1.1, side_offset=0.0, image_source='hand_color_image', global_orientation=True)
                    if retval == 0:
                        print("Stopping program...")
                        return
                    
                    time.sleep(1.0)
                    print("Turning to container")
                    self.turn_body_45()

                    print("Aligning gripper")
                    retval = self.align_to_shelf_location(container)
                    
                    if retval == 0:
                        self.tuck_arm()
                        print("Failed to align to shelf position, retrying...")
                        if align_attempts > 0:
                            print("Failed at aligning multiple times!")
                            return
                        else:
                            align_attempts += 1
                    else:
                        print("Localizing container")
                        retval, pos, rot, appr = self.localize_container()
                        if retval == 0:
                            print("Stopping program...")
                            return

                        #print("BOX POS:", pos)
                        #print("BOX POS IN WORLD", self.pose_in_body_to_world(pos))
                        
                        time.sleep(0.25)
                        print("Grasping container")
                        retval = self.grasp_container(pos, rot, appr)
                        if retval == 1:
                            container_grasped = True
                            print("Moving to trolley")
                            retval = self.move_to_trolley(trolley, 0.75)
                            if retval == 0:
                                print("Stopping program...")
                                return

                            print("Placing container on trolley")
                            retval = self.place_on_trolley(i)
                            if retval == 0:
                                print("Stopping program...")
                                return

                            print("Tucking arm back in")
                            self.tuck_arm()

                            i += 1

                        else:
                            self.tuck_arm()
                            self.open_gripper(0.0)
                            if grasp_attempts > 0:
                                print("Failed at grasping the container multiple times!")
                                return
                            else:
                                grasp_attempts += 1





    def test_task(self):
        shelf = 1
        trolley = 2

        #containers = []
        #print("Input container 1:")
        #containers.append(int(input()))

        #print("Input container 2:")
        #containers.append(int(input()))
        #containers = [12, 22, 32, 11, 21, 31]
        containers = [22]
        
        approach_pos = [0.993-0.25, 0.172-0.25, 0.268+0.05]
        approach_rot_q = math_helpers.Quat.from_yaw(math.pi/4)
        approach_rot = [approach_rot_q.w, approach_rot_q.x, approach_rot_q.y, approach_rot_q.z]

        lift_pos = [0.993, 0.172, 0.268+0.05]
        lift_rot_q = math_helpers.Quat.from_yaw(math.pi/4)
        lift_rot = [lift_rot_q.w, lift_rot_q.x, lift_rot_q.y, lift_rot_q.z]

        target_pos = [0.993, 0.172, 0.268+0.02]
        target_rot_q = math_helpers.Quat.from_yaw(math.pi/4)
        target_rot = [target_rot_q.w, target_rot_q.x, target_rot_q.y, target_rot_q.z]

        target = math_helpers.SE3Pose(x=target_pos[0], y=target_pos[1], z=target_pos[2], rot=math_helpers.Quat(w=target_rot[0], x=target_rot[1], y=target_rot[2], z=target_rot[3]))

        debug = False
        if not debug:
            i = 0
            for container in containers:
                container_grasped = False
                grasp_attempts = 0
                align_attempts = 0
                while not container_grasped:
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return

                    holding_box = False
                    while not holding_box:
                        self.open_gripper(0.0)
                        time.sleep(5)
                        self.open_gripper(1.0)
                        print("Enter 1 for succes, enter 0 to try again")
                        feedback = int(input())
                        if feedback == 1:
                            holding_box = True
                        else:
                            pass


                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return
                
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body', update_global=True)
                    if retval == 0:
                        print("Stopping program...")
                        return
                    
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, 1.25, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return

                    print("Approaching container")
                    # Approach shelf location
                    time.sleep(0.5)
                    retval = self.move_to_aruco(container, self.aruco_grid_size, dist_offset=1.1, side_offset=0.0, image_source='hand_color_image', global_orientation=True)
                    if retval == 0:
                        print("Stopping program...")
                        return
                    
                    time.sleep(1.0)
                    print("Turning to container")
                    self.turn_body_45()


                    self.movel(approach_pos, approach_rot, 5)
                    time.sleep(7)
                    self.movel(lift_pos, lift_rot, 5)
                    time.sleep(7)
                    self.movel(target_pos, target_rot, 5)
                    time.sleep(5)
                    self.test_log.write("### Ground truth ###\n")
                    target_in_world = self.pose_in_body_to_world(target)
                    self.test_log.write("Pose: {}\n\n".format(target_in_world))
                    self.open_gripper(0.0)
                    time.sleep(1)
                    self.movel(approach_pos, approach_rot, 2.5)
                    time.sleep(2.5)

                    self.tuck_arm()

                    self.test_task2()


    def test_task2(self):
        shelf = 1
        trolley = 2
        containers = [22, 22, 22, 22, 22]
        
        debug = False
        if not debug:
            i = 0
            for container in containers:
                container_grasped = False
                grasp_attempts = 0
                align_attempts = 0
                while not container_grasped:
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return

                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return
                
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, dist_offset=1.25, side_offset=0.5, image_source='body', update_global=True)
                    if retval == 0:
                        print("Stopping program...")
                        return
                    
                    print("Approaching shelf")
                    # Approach shelf
                    time.sleep(0.25)
                    retval = self.move_to_aruco(shelf, self.aruco_shelf_size, 1.25, image_source='body')
                    if retval == 0:
                        print("Stopping program...")
                        return

                    print("Approaching container")
                    # Approach shelf location
                    time.sleep(0.5)
                    retval = self.move_to_aruco(container, self.aruco_grid_size, dist_offset=1.1, side_offset=0.0, image_source='hand_color_image', global_orientation=True)
                    if retval == 0:
                        print("Stopping program...")
                        return
                    
                    time.sleep(1.0)
                    print("Turning to container")
                    self.turn_body_45()

                    print("Aligning gripper")
                    retval = self.align_to_shelf_location(container)
                    
                    if retval == 0:
                        self.tuck_arm()
                        print("Failed to align to shelf position, retrying...")
                        if align_attempts > 0:
                            print("Failed at aligning multiple times!")
                            return
                        else:
                            align_attempts += 1
                    else:
                        print("Localizing container")
                        retval, pos, rot, appr = self.localize_container()
                        if retval == 0:
                            print("Stopping program...")
                            return

                        approach = [appr.x, appr.y, appr.z]
                        self.movel(approach, rot, 3)
                        time.sleep(3.5)

                        log_pose = math_helpers.SE3Pose(x=pos.x, y=pos.y, z=pos.z, rot=math_helpers.Quat(w=rot[0], x=rot[1], y=rot[2], z=rot[3]))
                        log_pose_in_world = self.pose_in_body_to_world(log_pose)
                        self.test_log.write("### TEST {} ###\n".format(i))
                        self.test_log.write("Pose in world: {}\n\n".format(log_pose_in_world))

                        container_grasped = True

                        self.tuck_arm()

                        i += 1

    def test_task_safetylayer(self):
        iterations = 10
        for i in range(iterations):
            print("TEST ITERATION {}".format(i))
            time.sleep(2)
            d, img = self.check_for_human()
            print(img.shape)
            if d == 0 and np.sum(img) == 0: # if d and img == 0, then no human were found from any sources
                save_path = self.safety_log_path+"Test{}_no_humans.png".format(i)
                cv2.imwrite(save_path, img)
                self.safety_log.write("Test {}: {} (no humans found)\n".format(i, d))
            elif d == 0 and np.sum(img) > 0: #if d = 0 and img = an image, then depth was bad within the bounding box
                save_path = self.safety_log_path+"Test{}_bad_depth.png".format(i)
                cv2.imwrite(save_path, img)
                self.safety_log.write("Test {}: {} (bad depth in bbox)\n".format(i, d))
            elif d > 0 and np.sum(img) > 0: # if d = a distance and img = an image, then a human was recognized and a depth was computed
                save_path = self.safety_log_path+"Test{}_success.png".format(i)
                cv2.imwrite(save_path, img)
                self.safety_log.write("Test {}: {}\n".format(i, d))
        


        

    #####################################################################
    ######################### MAIN PROGRAMS #############################
    #####################################################################

    def main(self):
        with bosdyn.client.lease.LeaseKeepAlive(self.motion_client, must_acquire=True, return_at_exit=True):
            self.init_spot()

            self.kitting_task()
            
            self.poweroff_spot()

K = Kitting(sys.argv[1:])
K.main()
