import cv2
import torch
import pose_estimator
from torch.multiprocessing import Pool, Process, set_start_method, Queue
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import numpy as np
import main
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ('127.0.0.1', 5555)

device = 0 if torch.cuda.is_available() else 'cpu'
print(device)

flag = 0

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

def depth(q0, q1):
    while True:
        points0 = q0.get()
        points1 = q1.get()
        if points1 is not None and points0 is not None:
            p3d = []
            for i in range(17):
                point0 = points0[:, i, :]
                point1 = points1[:, i, :]
                point0 = point0.flatten()
                point1 = point1.flatten()
                _p3d = DLT(point0, point1)
                p3d.append(_p3d)
            points3D = np.array(p3d)
            with np.printoptions(precision = 3, suppress = True):
                sock.sendto(str.encode(str(points3D.tolist())), serverAddressPort)
def DLT(point0, point1):
    A = [point0[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point0[0] * P1[2, :],
        point1[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point1[0] * P2[2, :]
        ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]

def gammaCorrection0(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def gammaCorrection1(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

# A thread that starts live-feed for each camera that the method gets called with
# The ID is used to determine the position of each camera
def getFeed(ID, q):
    config_file = 'rtmpose-t_8xb256-420e_coco-256x192.py'
    checkpoint_file = 'rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth'
    model = init_model(config_file, checkpoint_file, device=0)
    cap = cv2.VideoCapture(ID)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, camera = cap.read()
        camera = cv2.GaussianBlur(camera, (5, 5), 0)
        try:
            # Calls the YOLOv8 pose estimation model and returns the needed keypoints
            if ID == 0:
                camera = gammaCorrection0(camera, 1.5)
                points = pose_estimator.framePose0(model, camera)
                if q.qsize() > 0:
                    q.get()
                q.put(points)
            else:
                camera = gammaCorrection1(camera, 1.5)
                points = pose_estimator.framePose1(model, camera)
                if q.qsize() > 0:
                    q.get()
                q.put(points)
            # Since the pose_estimator returns Null if there are no predications, and we know
            # that if there are predictions, then it will be of 17 points. We can directly
            # iterate for 17
            for i in range(17):
                # This takes each rows, makes it an array, and then uses each array for a circle
                point = points[:, i, :]
                point = point.flatten()
                cv2.circle(camera, (int(point[0]), int(point[1])), 5, (0,0,255), 3)
        except:
            None
        if ID == 0:
            name = "Cam 0"
        else:
            name = "Cam 1"
        cv2.imshow(name, camera)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Each existing camera gets a thread for its own, this will ensure that both cameeras will
    # work synchronously. This will decrease lag, and increase the accuracy.
    set_start_method('spawn')

    q0 = Queue()
    q1 = Queue()
    cam0 = Process(target=getFeed, args=(0, q0,))
    cam1 = Process(target=getFeed, args=(2, q1,))
    depthProcess = Process(target=depth, args=(q0, q1))

    cam0.start()
    cam1.start()
    depthProcess.start()

    cam0.join()
    cam1.join()
    depthProcess.join()
