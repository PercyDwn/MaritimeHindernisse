 ############test##################
#from ObjectHandler import ObjectHandler
from ObjectHandler import *

########
import  matplotlib.pyplot as plt
import numpy as np
import pyrate_sense_filters_gmphd
import random
from pyrate_common_math_gaussian import Gaussian
from pyrate_sense_filters_gmphd import GaussianMixturePHD
from numpy import vstack
from numpy import array
from numpy import ndarray
from numpy import eye
import math
import cv2

from phd_plot import *
from plotGMM import *
from boatMove import *

# Typing
from typing import List, Tuple

kmax = 50

#Red Boat location:
######################
#start
startRed = [[61.368],
            [-38.401], 
            [0.1798]]
#end
endRed = [[55.363],
            [-43.401], 
            [0.1798]]
redPos = BoatPos(startRed, endRed, 5, kmax)

#Green Boat location:
########################
#start
startGreen = [[59.363],
            [-49.401], 
            [0.1798]]
#end
endGreen = [[50.363],
            [-34.401], 
            [0.1798]]
greenPos = BoatPos(startGreen, endGreen, 5, kmax)

#Violet Boat location:
########################
#start
startVio = [[52.363],
            [-32.401], 
            [0.1798]]
#end
endVio = [[62.363],
          [-36.401], 
          [0.1798]]
vioPos = BoatPos(startVio, endVio, 5, kmax)

#Camera location:
#x = 53.162m
#y = -27.496m
#z = 0.7986m

camPos = [[53.162],
          [-27.496],
          [0.7986]]

for k in range(kmax):
    plt.plot(greenPos[k][0], greenPos[k][1],'ro',color= 'green', ms= 3)
    plt.plot(redPos[k][0], redPos[k][1],'ro',color= 'red', ms= 3)
    plt.plot(vioPos[k][0], vioPos[k][1],'ro',color= 'violet', ms= 3)

plt.plot(camPos[0], camPos[1], 'ro',color= 'black', ms= 5)
plt.show()

#Violet Boat location:
########################
#start
startVio = [[52.363],
            [-32.401], 
            [0.1798]]
#end
endVio = [[62.363],
          [-36.401], 
          [0.1798]]
alpha = math.atan(np.subtract(endVio[1],startVio[1])/np.subtract(endVio[0],startVio[0]))
#startMid
startVioMid = [[52.363],
            [-32.401], 
            [0.1798]]
#endMid
endVioMid = [[62.363],
          [-36.401], 
          [0.1798]]
vioMidPos = BoatPos(startVioMid, endVioMid, 5, kmax)