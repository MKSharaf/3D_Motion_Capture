import cv2
import numpy as np
import glob

checkerboardSize = (8,6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgLpoints = [] # 2d points in the left image plane.
imgRpoints = [] # 2d points in the right image plane.

imagesLeft = glob.glob('images/leftcam/*.png')
imagesRight = glob.glob('images/rightcam/*.png')

testImageL = cv2.imread(imagesLeft[0])
testImageR = cv2.imread(imagesRight[0])

for imageLeft, imageRight in zip(imagesLeft, imagesRight):
    imgL = cv2.imread(imageLeft)
    imgR = cv2.imread(imageRight)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, checkerboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, checkerboardSize, None)

    if retL and retR == True:

        objpoints.append(objp)
        cornersL = cv2.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(grayR,cornersR, (11,11), (-1,-1), criteria)

        imgLpoints.append(cornersL)
        imgRpoints.append(cornersR)

        cv2.drawChessboardCorners(imgL, checkerboardSize, cornersL, retL)
        cv2.drawChessboardCorners(imgR, checkerboardSize, cornersR, retR)

        cv2.imshow('Left Image', imgL)
        cv2.imshow('Right Image', imgR)

        cv2.waitKey(1000)

cv2.destroyAllWindows()

retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgLpoints, grayL.shape[::-1], None, None)
heightL, widthL, channelsL = imgL.shape
newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (widthL,heightL), 1, (widthL,heightL))

print("mtxL: ", mtxL)
print("newMtxL: ", newcameramtxL)

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgRpoints, grayR.shape[::-1], None, None)
heightR, widthR, channelsR = imgR.shape
newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (widthR,heightR), 1, (widthR,heightR))

print("mtxR: ", mtxR)
print("newMtxR: ", newcameramtxR)

# Stereo Calibration

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retStereo, newcameramtxL, distL, newcameramtxR, distR, rot, trans, essentialMtx, fundamentalMtx = cv2.stereoCalibrate(objpoints, imgLpoints, imgRpoints, newcameramtxL, distL, newcameramtxR, distR, grayL.shape[::-1], criteria, flags)

parameters = cv2.FileStorage('Parameters.xml', cv2.FILE_STORAGE_WRITE)

parameters.write('cameraMatrix_l', newcameramtxL)
parameters.write('cameraMatrix_r', newcameramtxR)
parameters.write('rotation', rot)
parameters.write('translation', trans)

print("Success")

parameters.release()
