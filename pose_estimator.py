import torch
from ultralytics import YOLO
import numpy as np
import queue
import cv2

# Setting up the pose estimation model
model = YOLO('YOLOv8s-pose.pt')
# Set the device to GPU if CUDA is installed and use the model with GPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(device)
def getKeypoints(results):
    for result in results:
        keypoint = result.keypoints.cpu().numpy()
        if np.shape(keypoint) == ((1, 17, 3)):
            xy_keypoints = np.delete(keypoint, np.s_[-1:], axis=2)
            return xy_keypoints
def framePose(id, frame):
    results = model.predict(source=frame, conf=0.75, device=device, show_labels=False, show_conf=False, boxes=False, verbose=False)
    if getKeypoints(results) is not None:
        points = getKeypoints(results)
        return points

