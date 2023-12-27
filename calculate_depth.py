import sys
import cv2
import numpy as np
import calculate_depth
import socket

# Initialize the socket to send to Unity
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server = ("127.0.0.1", 6666)

# Initialize the 2 points
left_points = None
right_points = None

# Get the remapping file
parameters = cv2.FileStorage()
parameters.open('Parameters.xml', cv2.FileStorage_READ)

# Initialize all the needed values to calculate depth
cameraMatrix_l = parameters.getNode('cameraMatrix_l').mat()
cameraMatrix_r = parameters.getNode('cameraMatrix_r').mat()
R = parameters.getNode('rotation').mat()
T = parameters.getNode('translation').mat()
points3D = []

# Get projection matrices
RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
P1 = cameraMatrix_l @ RT1

RT2 = np.concatenate([R, T], axis = -1)
P2 = cameraMatrix_r @ RT2

def get_point_left(points):
    calculate_depth.left_points = points

def get_point_right(points):
    calculate_depth.right_points = points

def depth():
    if calculate_depth.right_points is not None and calculate_depth.left_points is not None:
        p3d = []
        for i in range(17):
            left_point = left_points[:, i, :]
            right_point = right_points[:, i, :]
            left_point = left_point.flatten()
            right_point = right_point.flatten()
            _p3d = DLT(left_point, right_point)
            p3d.append(_p3d)
        calculate_depth.points3D = np.array(p3d)
        print(calculate_depth.points3D)
        #sock.sendto(str.encode(str(calculate_depth.points3D)), server)

def DLT(left_point, right_point):
    A = [left_point[1] * P1[2, :] - P1[1, :],
         P1[0, :] - left_point[0] * P1[2, :],
         right_point[1] * P2[2, :] - P2[1, :],
         P2[0, :] - right_point[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]
