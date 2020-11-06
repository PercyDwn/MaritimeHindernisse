import cv2
import numpy as np
import os
from ObjectHandler import *
from boatMove import *
from cv_transformation import *
from trans2d3d import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

res_x, res_y = 640, 480
# res_x, res_y = 1920, 1080
sens_w, sens_h = 36,20
fx = (50*res_x)/sens_w
fy = (50*res_y)/sens_h
cx = (res_x-1)/2
cy = (res_y-1)/2
M = np.array(
  [[fx, 0,  cx], 
   [0,  fy, cy], 
   [0,  0,  1]])
   # quadrions: x,y,z,w
r = Rot.from_quat([0.154, -0.640, -0.732, 0.176])

R = r.inv().as_matrix()
R = np.diag([1, -1, -1]).dot(R)
r = Rot.from_matrix(R)
print(R)
# R = inv(R)
# try R inv
t = np.vstack([-53.162, +27.496, -0.7986])
# t = np.vstack([53.162, -27.496, 0.7986])

Z_const = .0

redBoat2DCoord = np.array([
  (164,134),
  (175,134),
  (185,134),
  (195,133),
  (205,133),
  (215,133),
  (227,133),
  (238,133),
  (248,132),
  (258,132)
])

greenBoat2DCoord = np.array([
  (480,121)
])

fps=5
kmax=50
redStart = np.array([[61.368],[-38.401],[0.1798]])
redEnd = np.array([[55.363],[-43.401],[0.1798]])
redBoatPos = BoatPos(redStart, redEnd, fps, kmax, shift=1.1)
redBoat3DCoordinates = []

greenStart = np.array([[59.363],[-49.401],[0.1798]])
greenEnd = np.array([[50.363],[-34.401],[0.1798]])
greenBoatPos = BoatPos(greenStart, greenEnd, fps, kmax, shift=1.1)
greenBoat3DCoordinates = []


#================= 3d to 2d ===============================================================#
""" # red image point for t=1 is 165,135
cv2ImgPt = cv2.projectPoints(redBoatPos[0], r.as_rotvec(), R.dot(t), M, np.zeros((5)))
print('object point (red, t=1)')
print(redBoatPos[0])
print('calculated image point (red, t=1)')
print(cv2ImgPt[0][0][0])
print('real image point (red, t=1)')
print(redBoat2DCoord[0])  """
#=========================================================================================#

#================= 2d to 3d red ===========================================================#
""" for i in range(0,10):
    uvPoint = np.array([[redBoat2DCoord[i][0]],[redBoat2DCoord[i][1]],[1]])
    P = get3Dcoordinates(M,R,uvPoint,R.dot(t),Z_const,s=None)
    redBoat3DCoordinates.append(P)

for i,pt in enumerate(redBoat3DCoordinates, start=0):
  plt_transformed = plt.scatter(pt[0], pt[1], c='#ff0000', label='Transformierte Bildpunkte')
  plt_real = plt.scatter(redBoatPos[i][0], redBoatPos[i][1], c='#00ff00', label='Tatsächliche Objektpunkte')

plt.xlabel('x-Koordinate [m]')
plt.ylabel('y-Koordinate [m]')
plt.title('Bewegung des roten Bootes im Welt-Koordinatensystem', wrap=True)
plt.legend(handles=[plt_transformed, plt_real], loc='best')
plt.show() """
#=========================================================================================#

#================= 2d to 3d green ========================================================#
for i in range(0,1):
    uvPoint = np.array([[greenBoat2DCoord[i][0]],[greenBoat2DCoord[i][1]],[1]])
    P = get3Dcoordinates(M,R,uvPoint,R.dot(t),Z_const,s=None)
    greenBoat3DCoordinates.append(P)

for i,pt in enumerate(greenBoat3DCoordinates, start=0):
  plt_transformed = plt.scatter(pt[0], pt[1], c='#ff0000', label='Transformierte Bildpunkte')

for i,pt in enumerate(greenBoatPos, start=0):
  plt_real = plt.scatter(greenBoatPos[i][0], greenBoatPos[i][1], c='#00ff00', label='Tatsächliche Objektpunkte')

# plot camera
  plt_camera = plt.scatter(53.162, -27.496, c='#000000', label='Camera Location')

plt.xlabel('x-Koordinate [m]')
plt.ylabel('y-Koordinate [m]')
plt.title('Bewegung des grünen Bootes im Welt-Koordinatensystem', wrap=True)
plt.legend(handles=[plt_transformed, plt_real, plt_camera], loc='best')
plt.show()

# load image frame
img = cv2.imread(os.getcwd() + '/TestFile/Projekt/Bilder/list1/2.jpg')
img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
img = cv2.circle(img, (greenBoat2DCoord[0][0],greenBoat2DCoord[0][1]), radius=2, color=(0, 0, 255), thickness=-1)
cv2.imshow('Frame 2', img)
cv2.waitKey(0)
#=========================================================================================#