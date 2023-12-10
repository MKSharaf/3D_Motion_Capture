import cv2
import pose_estimator
import threading
import numpy as np

# Initializing the two cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Setting up the resolution to be 640x480 because it is the optimal resolution for YOLOv8
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

# A thread that starts live-feed for each camera that the method gets called with
# The ID is used to determine the position of each camera
def getFeed(cap, ID):
    while True:
        ret, camera = cap.read()
        camera = gammaCorrection(camera, 1.5)
        camera = cv2.GaussianBlur(camera, (7, 7), 0)
        try:
            # Calls the YOLOv8 pose estimation model and returns the needed keypoints
            points = pose_estimator.framePose(0, camera)

            # Since the pose_estimator returns Null if there are no predications, and we know
            # that if there are predictions, then it will be of 17 points. We can directly
            # iterate for 17
            for i in range(17):
                # This takes each rows, makes it an array, and then uses each array for a circle
                point = points[:, i, :]
                point = point.flatten()
                cv2.circle(camera, (int(point[0]), int(point[1])), 5, (0,0,255), 3)
        except:
            print()
        if ID == 0:
            name = "Left Cam"
        else:
            name = "Right Cam"
        cv2.imshow(name, camera)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Each existing camera gets a thread for its own, this will ensure that both cameeras will
    # work synchronously. This will decrease lag, and increase the accuracy.
    cam0 = threading.Thread(target=getFeed, args=(cap0, 0,))
    cam1 = threading.Thread(target=getFeed, args=(cap1, 1,))

    cam0.start()
    cam1.start()