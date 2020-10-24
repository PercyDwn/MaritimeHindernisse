import cv2
import numpy as np
import os
from ObjectHandler import *

def line_eq(p1: tuple, p2: tuple ):
    # points come as (x,y)
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p2[1] - (m*p2[0])

    return m,b

def line_gen_eq(m,b):
    return -m, 1, -b

def get_d(A,B,C,u,v) -> float:
    return abs(A*u+B*v+C) / sqrt(A**2 + B**2)

def get_Z(f,d,H) -> float:
    return (f/d)*H

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)
ObjectHandler.setPlotOnUpdate(False)

H = 0.7986
f = 2666.67

for t in range(1,21):
    print("timestep %d" % t)
    Objects = ObjectHandler.getObjectStates(t = t)
    DetectedObstacles = ObjectHandler.getDetectedObstacles(t=t)
    HorizontLines = ObjectHandler.getHorizonLines(t = t)
    Distances = []
    for i, obj in enumerate(Objects, start=1):
      u = obj[0]
      v = obj[1]
      for li, line in enumerate(HorizontLines, start=1):
          m,b = line_eq(line.end_points[0], line.end_points[1])
          A,B,C = line_gen_eq(m,b)
          d = get_d(A,B,C,u,v)
          Z = get_Z(f,d,H)
          if li == 1: Distances.append(Z)
          print("Object #%d (%d,%d) is %.1f m away" % (i,u,v,Z))
    
    img = cv2.imread(os.getcwd()+'/TestFile/Projekt/Bilder/list1/'+str(t)+'.jpg')
    ObjectHandler.plot_distances(img, DetectedObstacles, HorizontLines, Distances)
    cv2.waitKey(0)
    print("---------")
