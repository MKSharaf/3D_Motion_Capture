import numpy as np
import cv2
import Yolov8

# Initializing the two cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Due to the bandwidth of typical USB webcameras, we will need to enhance the images captured from the 2nd camera

# Initializing the alpha, and beta for the 2nd camera

alpha = 2
beta = 20

while True:
    ret0, cameraL = cap0.read()
    ret1, cameraR = cap1.read()
    if (ret0):
        try:
            points = Yolov8.framePose(0, cameraL)
            for i in range(17):
                point = points[:, i, :]
                point = point.flatten()
                if point[0] and point[1] != 0:
                    cv2.circle(cameraL, (int(point[0]), int(point[1])), 5, (0,0,255), 3)
        except:
            print("")
        cv2.imshow("Camera Left", cameraL)
    if (ret1):
        # Making the image brighter, and also increasing contrast
        Fixed_cameraR = cv2.convertScaleAbs(cameraR, alpha=alpha, beta=beta)
        try:
            points = Yolov8.framePose(1, Fixed_cameraR)
            for i in range(17):
                point = points[:, i, :]
                point = point.flatten()
                if point[0] and point[1] != 0:
                    cv2.circle(Fixed_cameraR, (int(point[0]), int(point[1])), 5, (0, 0, 255), 3)
        except:
            print("")
        cv2.imshow("Camera Right", Fixed_cameraR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()