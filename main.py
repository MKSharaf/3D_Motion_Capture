import numpy as np
import cv2
import pose_estimator
import threading

# Initializing the two cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# A thread that starts live-feed for each camera that the method gets called with
# The ID is used to determine the position of each camera
def getFeed(cap, ID):
    while True:
        ret, camera = cap.read()
        try:
            # Calls the Yolov8 pose estimation model and returns the needed keypoints
            points = pose_estimator.framePose(0, camera)

            # Since the pose_estimator returns Null if there are no predications, and we know
            # that if there are predictions, then it will be of 17 points. We can directly
            # iterate for 17
            for i in range(17):
                # This takes each rows, makes it an array, and then uses each array for a circle
                point = points[:, i, :]
                point = point.flatten()
                if point[0] and point[1] != 0:
                    cv2.circle(camera, (int(point[0]), int(point[1])), 5, (0,0,255), 3)
        except:
            print()
        name = ""
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