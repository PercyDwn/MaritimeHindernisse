import cv2
import numpy as np
import os
from ObjectHandler import *
from boatMove import *
from cv_transformation import *
from trans2d3d import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot



fx = 50/36*1920
fy = 50/24*1080
cx = 1920/2
cy = 1080/2
M = np.array(
  [[fx, 0,  cx], 
   [0,  fy, cy], 
   [0,  0,  1]])
r = Rot.from_quat([0.154, -0.640, -0.732, 0.176])
R = r.as_matrix()
t = np.vstack([-53.162, +27.496, -0.7986])

Z_const = .1798

redBoat2DCoord = np.array([
  (165,135),
  (173,133),
  (184,131),
  (193,131),
  (204,132),
  (215,134),
  (227,132),
  (238,131),
  (247,130),
  (256,131)
])

fps=5
kmax=50
redStart = np.array([[61.368],[-38.401],[0.1798]])
redEnd = np.array([[55.363],[-43.401],[0.1798]])
redBoatPos = BoatPos(redStart, redEnd, fps, kmax)
redBoat3DCoordinates = []

""" img = cv2.imread(os.getcwd()+'/TestFile/Projekt/Bilder/list1/1.jpg')
cv2.imshow('Name',img)
cv2.waitKey(0) """

cv2ImgPt = cv2.projectPoints(redBoatPos[0], r.as_rotvec(), t, M, np.zeros((5)))
print(cv2ImgPt[0][0][0])


""" for i in range(0,10):
    uvPoint = np.array([[redBoat2DCoord[i][0]],[redBoat2DCoord[i][1]],[1]])
    P = get3Dcoordinates(M,R,uvPoint,t,Z_const,s=None)
    redBoat3DCoordinates.append(P)
    print("calculated point:")
    print(P)
    print("actual point:")
    print(redBoatPos[i])

for i,pt in enumerate(redBoat3DCoordinates, start=0):
  plt.scatter(pt[0], pt[1], c='#ff0000')
  plt.scatter(redBoatPos[i][0], redBoatPos[i][1], c='#00ff00')
plt.show() """
