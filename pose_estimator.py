import numpy as np
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

def getKeypoints0(results):
    for result in results:
        # Extracts the keypoints from the predications
        keypoint =  results[0].pred_instances
        if np.shape(keypoint.keypoints) == ((1, 17, 2)):
            return keypoint.keypoints

def framePose0(model, frame):
    # This initializes the YOLOv8 model to use it to predict the keypoints in each frame
    results = inference_topdown(model, frame)
    # We make sure that we are sending a value back to the cameras
    if getKeypoints0(results) is not None:
        points = getKeypoints0(results)
        return points
        
def getKeypoints1(results):
    for result in results:
        # Extracts the keypoints from the predications
        keypoint =  results[0].pred_instances
        if np.shape(keypoint.keypoints) == ((1, 17, 2)):
            return keypoint.keypoints

def framePose1(model, frame):
    # This initializes the YOLOv8 model to use it to predict the keypoints in each frame
    results = inference_topdown(model, frame)
    # We make sure that we are sending a value back to the cameras
    if getKeypoints1(results) is not None:
        points = getKeypoints1(results)
        return points
