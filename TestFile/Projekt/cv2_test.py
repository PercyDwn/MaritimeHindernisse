import cv2
import numpy as np
import glob

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

objpoints = []
imgpoints = []
images = glob.glob('TestFile/Projekt/Bilder/cv2/*3.jpg')

for fname in images:
  img = cv2.imread(fname)
  diceimg = cv2.imread(fname)
  grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  found, corners = cv2.findChessboardCorners(grayimg,(7,7), None)

  if found == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(grayimg, corners, (11,11), (-1, -1), criteria)
    imgpoints.append(corners)

    cv2.drawChessboardCorners(img, (7,7), corners2, found)
    cv2.imshow('chessboard', img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayimg.shape[::-1], None, None)
  else:
    print('corners not found')

  corners2 = cv2.cornerSubPix(grayimg,corners,(11,11),(-1,-1),criteria)
  _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
  axis = np.float32( [ [0,0,0], [0,5,0], [5,5,0], [5,0,0], [0,0,-3],[0,5,-3],[5,5,-3],[5,0,-3] ] )
  imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

  # draw(img, corners, imgpts)
  # cv2.imshow("cube", img)
  # cv2.waitKey(0)

  # print(imgpts)
  # print("...")
  for pt in imgpts:
    cv2.circle(diceimg, (pt[0][0], pt[0][1]), radius=1, color=(0,0,255), thickness=-1)

  p1 = imgpts[0][0]
  p2 = imgpts[1][0]
  p3 = imgpts[2][0]
  p4 = imgpts[3][0]
  p5 = imgpts[4][0]
  p6 = imgpts[5][0]
  p7 = imgpts[6][0]
  p8 = imgpts[7][0]

  linewidth = 2

  cv2.line(diceimg, tuple(p1), tuple(p2), (0,0,255),linewidth)
  cv2.line(diceimg, tuple(p2), tuple(p3), (0,0,255),linewidth)
  cv2.line(diceimg, tuple(p3), tuple(p4), (0,0,255),linewidth)
  cv2.line(diceimg, tuple(p4), tuple(p1), (0,0,255),linewidth)

  cv2.line(diceimg, tuple(p5), tuple(p6), (0,255,0),linewidth)
  cv2.line(diceimg, tuple(p6), tuple(p7), (0,255,0),linewidth)
  cv2.line(diceimg, tuple(p7), tuple(p8), (0,255,0),linewidth)
  cv2.line(diceimg, tuple(p8), tuple(p5), (0,255,0),linewidth)

  cv2.line(diceimg, tuple(p1), tuple(p5), (255,0,0),linewidth)
  cv2.line(diceimg, tuple(p2), tuple(p6), (255,0,0),linewidth)
  cv2.line(diceimg, tuple(p3), tuple(p7), (255,0,0),linewidth)
  cv2.line(diceimg, tuple(p4), tuple(p8), (255,0,0),linewidth)

  cv2.imshow("chessboard with dots", diceimg)

  print("camera matrix:")
  print(mtx)
  print("rotation matrix:")
  print(rvecs)
  print("translation vector:")
  print(tvecs)
  print("distortion:")
  print(dist) 

  cv2.waitKey(0)
 
  cv2.destroyAllWindows()