# AutoKit - A General Robot System for Autonomous Batch Kitting


## Background

The goal of AutoKit is to automate the process of kitting for batch production, specifically developed for assembly of device components at Novo Nordisk. Traditional manual kitting methods can be time-consuming and induce ergonomic stress. AutoKit aims to automate this process by leveraging advanced perception, navigation and manipulation techniques.

## Hardware

To use AutoKit directly, the following hardware components are required:

1. **Spot robot with arm** (With SDK version 3.1.1)
2. **Container end-effector** (3D printable custom end-effector can be found at:)
3. **Computer** (With specs sufficient to run YOLOv5)
4. **Storage rack** (Front bottom must be indicated by an ArUco marker, shelf locations must be indicated by individual ArUco markers)
5. **Trolley** (Front bottom must be indicated by an ArUco marker)
6. **Containers** (Box-like in appearance and placed on the storage rack)

Please ensure that you have the necessary hardware components before proceeding with the installation.

## Installation

To install and set up the AutoKit system, follow these steps:

1. **Clone the Repository**: Start by cloning the AutoKit repository from GitHub to your local machine.
git clone https://github.com/your-username/AutoKit.git

2. **Install Dependencies**: Navigate to the cloned repository and install the required dependencies using the package manager of your choice. For example, if you are using Python and pip, run the following command:
pip install -r requirements.txt

3. **Run AutoKit**: Once the dependencies are installed and the hardware is set up, you can run the AutoKit system. Execute the main script, typically named `autokit.py`, to start the application.
python autokit.py

4. **Additional Configuration**: If necessary, you can modify e.g., ArUco dictionaries or placing positions on the trolley to customize the behavior of the AutoKit system. Refer to the documentation or comments within the codebase for guidance on configuration options.

That's it! You have successfully installed and set up the AutoKit system. Feel free to explore the different features and functionalities provided by the system.

## Contact

For any questions or inquiries, please reach out at philip_lund_moller@hotmail.com.
