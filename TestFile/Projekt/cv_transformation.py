import cv2
import numpy as np
from numpy.linalg import inv
import glob
from scipy.spatial.transform import Rotation as Rot



""" h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
cv2.waitKey(0) """


def initCam(images, cbw, cbh):
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  objp = np.zeros((cbw*cbh,3), np.float32)
  objp[:,:2] = np.mgrid[0:cbw,0:cbh].T.reshape(-1,2)

  objpoints = []
  imgpoints = []
  for fname in images:
    img = cv2.imread(fname)
    point_img = cv2.imread(fname)
    diceimg = cv2.imread(fname)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(grayimg,(cbw,cbh), None)

    if found == True:
      objpoints.append(objp)
      corners2 = cv2.cornerSubPix(grayimg, corners, (11,11), (-1, -1), criteria)
      imgpoints.append(corners)

      cv2.drawChessboardCorners(img, (cbw,cbh), corners2, found)
      # cv2.imshow('chessboard', img)

      ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayimg.shape[::-1], None, None)
    else:
      print('corners not found')

    corners2 = cv2.cornerSubPix(grayimg,corners,(11,11),(-1,-1),criteria)
    ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    drawCube(img, corners, imgpts)
  
  return img, mtx, dist

def undistort(mtx, dist, img):
  h,  w = img.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
  print("old camera matrix:")
  print(mtx)
  print("new camera matrix:")
  print(newcameramtx)
  # undistort
  mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
  dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

  # crop the image
  x,y,w,h = roi
  dst = dst[y:y+h, x:x+w]
  cv2.imwrite('calibresult.png',dst)

  return newcameramtx

def drawCube(img, corners, imgpts) -> None:
  imgpts = np.int32(imgpts).reshape(-1,2)

  # draw ground floor in green
  img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

  # draw pillars in blue color
  for i,j in zip(range(4),range(4,8)):
      img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

  # draw top layer in red color
  img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

  cv2.imshow("Chessboard Cube", img)
  cv2.waitKey(0)

def transform2DPoint(mtx, dist, objpoints, imgpoints, uvPoint, point_img) -> None:

    for pt in imgpoints:
      cv2.circle(point_img, (pt[0][0], pt[0][1]), radius=1, color=(0,0,255), thickness=3)
    cv2.imshow("Points in Room", point_img)
    cv2.waitKey(0)

    ret, rvecs, tvecs = cv2.solvePnP(objpoints, imgpoints, mtx, dist)

    rvecs_t = np.array( [rvecs[0][0], rvecs[1][0], rvecs[2][0]] )
    Rot_Obj = Rot.from_rotvec(rvecs_t)
    R = Rot_Obj.as_matrix()

    # inv(R)@inv(M)@s*(u,v,1) = (X,Y,Z_const)+inv(R)@t
    left = inv(R) @ inv(mtx) @ uvPoint
    right = inv(R) @ tvecs

    s = (75 + right[2][0]) / (left[2][0])
    P = inv(R) @ (s * inv(mtx) @ uvPoint - tvecs) 

    print("s:" + str(s))

    print("Point:")
    print(P)
  

images = glob.glob('TestFile/Projekt/Bilder/cv2/room_201020.jpg')
# img points for 1920 image
objpoints = np.float32([[0,0,0],[1010,400,0],[145,1165,0],[880,1190,0]])
imgpoints = np.float32([[[1780,1333]],[[378,732]],[[1360,267]],[[730,221]]])
uvPoint = np.array([[882],[459],[1]])
cbw = 9
cbh = 6

""" images = glob.glob('TestFile/Projekt/Bilder/cv2/room_201020_4032.jpg')
# img points for 4032 image
objpoints = np.float32([[0,0,0],[1420,400,0],[145,1165,0],[870,1220,0]])
imgpoints = np.float32([[[3740,2802]],[[795,1538]],[[2856,560]],[[1534,463]]])
uvPoint = np.array([[1852],[963],[1]])
cbw = 9
cbh = 6 """

img, mtx, dist = initCam(images, cbw, cbh)
newcameramtx = undistort(mtx, dist, img)
transform2DPoint(newcameramtx, dist, objpoints, imgpoints, uvPoint, img)

""" images = glob.glob('TestFile/Projekt/Bilder/cv2/chessboard_3.jpg')
cbw = 7
cbh = 7

img, mtx, dist = initCam(images, cbw, cbh)
undistort(mtx, dist, img) """
# transform2DPoint(mtx, dist, objpoints, imgpoints, img)

cv2.destroyAllWindows()

"""
Todo:
- improve intrinsic camera values
"""