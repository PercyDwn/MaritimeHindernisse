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

