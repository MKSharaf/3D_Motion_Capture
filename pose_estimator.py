import torch
from ultralytics import YOLO
import numpy as np

# Setting up the pose estimation model
model = YOLO('YOLOv8m-pose.pt')
# Set the device to GPU if CUDA is installed and use the model with GPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(device)

def getKeypointsLeft(results):
    for result in results:
        # Extracts the keypoints from the predications
        keypoint = result.keypoints.cpu().numpy()
        if np.shape(keypoint) == ((1, 17, 3)):
            # Removes the final column because so far we only need x and y
            xy_keypoints = np.delete(keypoint, np.s_[-1:], axis=2)
            return xy_keypoints

def framePoseLeft(frame):
    # This initializes the YOLOv8 model to use it to predict the keypoints in each frame
    results = model.predict(source=frame, conf=0.5, device=device, show_labels=False, show_conf=False, boxes=False, verbose=False)
    # We make sure that we are sending a value back to the cameras
    if getKeypointsLeft(results) is not None:
        points = getKeypointsLeft(results)
        return points
def getKeypointsRight(results):
    for result in results:
        # Extracts the keypoints from the predications
        keypoint = result.keypoints.cpu().numpy()
        if np.shape(keypoint) == ((1, 17, 3)):
            # Removes the final column because so far we only need x and y
            xy_keypoints = np.delete(keypoint, np.s_[-1:], axis=2)
            return xy_keypoints

def framePoseRight(frame):
    # This initializes the YOLOv8 model to use it to predict the keypoints in each frame
    results = model.predict(source=frame, conf=0.8, device=device, show_labels=False, show_conf=False, boxes=False, verbose=False)
    # We make sure that we are sending a value back to the cameras
    if getKeypointsRight(results) is not None:
        points = getKeypointsRight(results)
        return points