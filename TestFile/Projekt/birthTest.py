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
from mpl_toolkits.mplot3d import Axes3D
import cv2

from phd_plot import *
from plotGMM import *

# Typing
from typing import List, Tuple


def phd_BirthModels (num_w: int, num_h: int) -> List[Gaussian]:
    """
     Args:
            ObjectHandler: ObjectHandler          
            num_w: number of fields on width
            num_h: number of Fields on height
            
    """

    # #ObjectHandler updaten
    ##--------------------------
    #k = 1   #Zeitschritt
    #pictures_availiable = True
    #fig = plt.figure()
    #est_phd: ndarray = []

    #while pictures_availiable == True: #While: 
    #    try:
    #        #ObjectHandler.updateObjectStates()
    #        current_measurement_k = ObjectHandler.getObjectStates(k) #Daten der Detektion eines Zeitschrittes 
    #        #print(current_measurement_k)
    #        #print([current_measurement_k[0][0]])
    #    except InvalidTimeStepError as e:
    #        print(e.args[0])
    #        k = 0                
    #        pictures_availiable = False
    #        break
    #    k = k+1
    ## Bild höhe und breite Abrufen
    ##obj_h = ObjectHandler.getImgHeight()
    ##obj_w = ObjectHandler.getImgWidth()

    obj_h = 480
    obj_w = 640

    birth_belief: List[Gaussian] = []

    # Birthmodelle Rand links
    #--------------------------
    b_leftside: List[Gaussian] = [] 
    cov_edge = array([[100*30,  0.0,             0.0, 0.0], 
                     [0.0,  (obj_h/(num_h))*30,   0.0, 0.0],
                     [0.0,  0.0,             10.0*10, 0.0],
                     [0.0,  0.0,             0.0, 10.0*10]])
    print('leftside')
    for i in range(num_h):
        mean = vstack([0,  i*obj_h/num_h+obj_h/(2*num_h), 10.0, 0.0])
        print(i*obj_h/num_h+obj_h/(2*num_h))
        print('--------------')
        b_leftside.append(Gaussian(mean, cov_edge, 0.05))
    
    # Birthmodelle Rand rechts
    #--------------------------
    b_rightside: List[Gaussian] = [] 
    print('*******************')
    print('rightside')
    for i in range(num_h):
        mean = vstack([obj_w,  i*obj_h/num_h+obj_h/(2*num_h), -10.0, 0.0])
        b_rightside.append(Gaussian(mean, cov_edge, 0.05))

 
    # Birthmodelle übers Bild
    #--------------------------
    cov_area = array([[(obj_w/num_w)*30, 0.0,            0.0,    0.0], 
                     [0.0,          (obj_h/(num_h))*30,  0.0,    0.0],
                     [0.0,          0.0,            10.0*10,   0.0],
                     [0.0,          0.0,            0.0,    10.0*10]])
    b_area: List[Gaussian] = []
    for i in range(num_h):
        for j in range(num_w): 
            mean = vstack([j*obj_w/num_w+obj_w/(2*num_w), i*obj_h/num_h+obj_h/(2*num_h), 0.0, 0.0])
            b_area.append(Gaussian(mean, cov_area, 0.1))
    
    
    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)
    birth_belief.extend(b_area)

    return birth_belief

for n in range(5):
    birth_belief = phd_BirthModels(n+1, n+1)

    fig = plt.figure
    fig = plotGMM(birth_belief, 640, 480, 3)
    plt.title('Gausplot für : ' +str(n+1))
    plt.show()