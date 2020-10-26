import cv2
import numpy as np
import os
from ObjectHandler import *
from boatMove import *
import matplotlib.pyplot as plt

def line_eq(p1: tuple, p2: tuple ):
    # points come as (x,y)
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p2[1] - (m*p2[0])
    return m,b

def line_gen_eq(m,b):
    return -m/2, 2, -b/2

def get_d(A,B,C,u,v) -> float:
    return abs(A*u+B*v+C) / sqrt(A**2 + B**2)

def get_Z(f,d,H) -> float:
    return (f/d)*H

def get_real_Z(p1, p2) -> float:
    return sqrt( ((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2) )

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)
ObjectHandler.setPlotOnUpdate(False)

H = 0.7986
f = 2666.67

kmax = 50
fps = 5

redStart = np.array([[61.368],[-38.401],[0.1798]])
redEnd = np.array([[55.363],[-43.401],[0.1798]])
greenStart = np.array([[59.363],[-49.401],[0.1798]])
greenEnd = np.array([[50.363],[-34.401],[0.1798]])
blueStart = np.array([[52.363],[-32.401],[0.1798]])
blueEnd = np.array([[62.363],[-36.401],[0.1798]])

camLocation = np.array([[53.162],[-27.496],[0.7986]])

redBoatPos = BoatPos(redStart, redEnd, fps, kmax)
greenBoatPos = BoatPos(greenStart, greenEnd, fps, kmax)
blueBoatPos = BoatPos(blueStart, blueEnd, fps, kmax)

""" plt.axis([50,65,-50,-20])
plt.scatter(*zip(*redBoatPos), c='#ff0000')
plt.scatter(*zip(*greenBoatPos), c='#00ff00')
plt.scatter(*zip(*blueBoatPos), c='#0000ff')
plt.scatter(x=53.162,y=-27.496, c='#000000')
plt.show() """

for t in range(1,21):
    print("timestep %d" % t)
    Objects = ObjectHandler.getObjectStates(t = t)
    DetectedObstacles = ObjectHandler.getDetectedObstacles(t=t)
    HorizontLines = ObjectHandler.getHorizonLines(t = t)
    Distances = []
    for i, obj in enumerate(Objects, start=1):
      u = obj[0]
      v = obj[1]
      Z_sum = 0
      Z_count = 0
      for li, line in enumerate(HorizontLines, start=1):
          m,b = line_eq(line.end_points[0], line.end_points[1])
          A,B,C = line_gen_eq(m,b)
          d = get_d(A,B,C,u,v)
          Z = get_Z(f,d,H)
          Z_sum += Z 
          Z_count += 1
      Distances.append( (Z_sum/Z_count) )
    
    realZredBoat = get_real_Z(redBoatPos[t-1], camLocation)
    realZgreenBoat = get_real_Z(greenBoatPos[t-1], camLocation)
    realZblueBoat = get_real_Z(blueBoatPos[t-1], camLocation)
    print("The red boat is %.1f m away" % (realZredBoat))
    print("The green boat is %.1f m away" % (realZgreenBoat))
    print("The blue boat is %.1f m away" % (realZblueBoat))

    # load img and plot obstacles and distances
    img = cv2.imread(os.getcwd()+'/TestFile/Projekt/Bilder/list1/'+str(t)+'.jpg')
    img = ObjectHandler.plot_distances(img, DetectedObstacles, HorizontLines, Distances)
    # ploat boat locations into img
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = .5
    fontColor              = (255,30,30)
    lineType               = 1
    position = (5, 15)
    text = "red: %.1f m | green: %.1f m | blue: %.1f m" % (realZredBoat,realZgreenBoat,realZblueBoat)
    cv2.putText(img,text,position,font,fontScale,fontColor,lineType)

    cv2.imshow('Image',img)
    cv2.waitKey(0)
    print("---------")
