from ObjectHandler import ObjectHandler

import  matplotlib.pyplot as plt
# Mathematics
from numpy.linalg import inv
import numpy as np
from numpy import array
import math
import cv2
from cv2 import Rodrigues

from scipy.spatial.transform import Rotation as Rot

def trans(u: int, v: int, fx: int, fy: int, cx: int, cy: int, R_quat: tuple, t: array) -> array:
    """
    Args:
        u: x Coordinates in Pixel
        v: y Coordinates in Pixel
        A: Camera Martrix / intrinsic matrix
        fx, fy: focal length in pixel units
        cx, cy: 
        R: rotian Matrix of the camera
        t: Translation Matrix of the camera
        [R|t]: joint rotation-translation matrix
       
    """
    v_pix = [[u],
             [v],
             [1]]
    A = [[fx, 0,  cx], 
         [0,  fy, cy], 
         [0,  0,  1]]
    # create Rotation object with given quatrions
    r = Rot.from_quat([R_quat[0], R_quat[1], R_quat[2], R_quat[3]])
    R = r.as_matrix()

    # Rt = np.hstack(R, t))

    # not working as Rt is not a square matrix
    # Vec = inv(A)@inv(Rt)@v_pix

    # not working either
    # a = A @ Rt
    # b = v_pix
    # x = np.linalg.lstsq(a=a,b=b, rcond=None)

    s = 1
    Vec = inv(R) @ inv(A) * s @ v_pix - inv(R) @ t

    return Vec

def initTrans(objectPoints: array, imagePoints: array, cameraMatrix: array) -> array:

    distCoeffs = np.array([])
    retval, rvec, tvec = cv2.solvePnP(objectPoints = objectPoints, imagePoints = imagePoints, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)

    return retval, rvev, tvec

def retrieveImagePoints() -> None:
    ObjectHandler = ObjectHandler()
    ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1')
    ObjectHandler.setImageBaseName('')
    ObjectHandler.setImageFileType('.jpg')
    ObjectHandler.setDebugLevel(2)
    ObjectHandler.setPlotOnUpdate(True)

    ObjectData = ObjectHandler.getObjectStates(t=1, position='cb')
    cv2.waitKey(0)


def trans3d2d(coord3D, cameraMatrix: array, rotMatrix: array, transVec: array) -> array:
    ObjPt = np.array([coord3D[0], coord3D[1], coord3D[2], [1]])
    Rt = np.hstack((rotMatrix, transVec))
    vec_2d = cameraMatrix @ Rt @ ObjPt
    return vec_2d

if __name__ == "__main__":
    u = 83
    v = 176
    fx = 2667
    fy = 2250
    cx = 1920/2
    cy = 1080/2
    R_quat = (0.176, 0.154, -0.640, -0.732)
    t = np.vstack([53.162, -27.496, 0.7986])
    r = Rot.from_quat([R_quat[0], R_quat[1], R_quat[2], R_quat[3]])

    Vec = trans(u=u, v=v, fx=fx, fy=fy, cx=cx, cy=cy, R_quat=R_quat, t=t)
    print(Vec)

    # camera matrix
    M = np.array(
      [[fx, 0,  cx], 
       [0,  fy, cy], 
       [0,  0,  1]])
    
    # not working
      # k_real = trans(u, v, fx, fy, cx, cy, R_quat, t)

    # retrieve imagePoints
    # retrieveImagePoints()
  
    # 3d object points: red boat and green boat
    objectPoints = np.zeros(shape=(2,3,1))
    #objectPoints = np.array([(61.368, -38.401, 0.1798),(59.363, -49.401, 0.1798)])
    objectRed = np.vstack([61.368, -38.401, 0.1798])
    objectGreen = np.vstack([59.363, -49.401, 0.1798])
    objectPoints[0] = objectRed;
    objectPoints[1] = objectGreen;
    # 2d image points: red boat and green boat
    imagePoints = np.array([(83,176),(471, 124)]).T

    imagePoints = np.zeros(shape=(2,2,1))
    #imagePoints = np.array([(83,176),(471, 124)])
    imageRed = np.vstack([83,176])
    imageGreen = np.vstack([471, 124])
    imagePoints[0] = imageRed;
    imagePoints[1] = imageGreen;

    trans3d2d(X=61.368,Y=-38.401,Z=0.1798, cameraMatrix=M, rotMatrix=r.as_matrix(), transVec=t)

    # retval,rvev,tvec = initTrans(objectPoints=objectPoints, imagePoints=imagePoints, cameraMatrix=M)
    
    #print(retval)
    #print(rvev)
    #print(tvec)

    