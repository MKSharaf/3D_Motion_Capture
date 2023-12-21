import cv2
import time

cap0 = cv2.VideoCapture(0)
cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap0.set(cv2.CAP_PROP_EXPOSURE, -6)
cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap1.set(cv2.CAP_PROP_EXPOSURE, -6)


num = 0
while (True):
    ret0, img0 = cap0.read()
    ret1, img1 = cap1.read()

    cv2.imshow("Left", img0)
    cv2.imshow("Right", img1)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('images/leftcam/leftimg' + str(num) + '.png', img0)
        cv2.imwrite('images/rightcam/rightimg' + str(num) + '.png', img1)
        print("Images Saved!")
        num+=1
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break