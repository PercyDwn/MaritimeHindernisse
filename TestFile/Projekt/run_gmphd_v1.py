# import stuff

import os

from ObjectHandler import ObjectHandler

import matplotlib.pyplot as plt

from pyrate_sense_filters_gmphd import *
from pyrate_common_math_gaussian import *
from phd_plot import *

import numpy as np
from numpy import vstack, array, ndarray, eye

import math 
import random

from mpl_toolkits.mplot3d import Axes3D

import cv2

from typing import List, Tuple

# init ObjectHandler
ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)
ObjectHandler.setPlotOnUpdate(False)

# init gmphd

F = array(
    [[1.0, 0.0, 1.0, 0.0], 
    [0.0, 1.0, 0.0, 1.0], 
    [0.0, 0.0, 1.0, 0.0], 
    [0.0, 0.0, 0.0, 1.0]]
)

H = array(
    [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]]
)

Q = 15*eye(4)
R = 35*eye(2)

# settings

survival_rate = 0.999
detection_rate = 0.9
intensity = 0.0001

# birth belief

def phd_BirthModels (num_w: int, num_h: int) -> List[Gaussian]:
    """
     Args:
            ObjectHandler: ObjectHandler          
            num_w: number of fields on width
            num_h: number of Fields on height
            
    """

    obj_h = 480
    obj_w = 640

    birth_belief: List[Gaussian] = []

    # Birthmodelle Rand links
    #--------------------------
    b_leftside: List[Gaussian] = [] 
    cov_edge = array([[20,  0.0,             0.0, 0.0], 
                     [0.0,  obj_h/(num_h),   0.0, 0.0],
                     [0.0,  0.0,             20.0, 0.0],
                     [0.0,  0.0,             0.0, 20.0]])
    for i in range(1,num_h):
        mean = vstack([20, i*obj_h/(num_h+1), 10.0, 0.0])
        b_leftside.append(Gaussian(mean, cov_edge))
    
    # Birthmodelle Rand rechts
    #--------------------------
    b_rightside: List[Gaussian] = [] 
    for i in range(1,num_h):
        mean = vstack([obj_w-20, i*obj_h/(num_h+1), -10.0, 0.0])
        b_rightside.append(Gaussian(mean, cov_edge))

 
    # Birthmodelle übers Bild
    #--------------------------
    cov_area = array([[obj_w/num_w, 0.0,            0.0,    0.0], 
                     [0.0,          obj_h/(num_h),  0.0,    0.0],
                     [0.0,          0.0,            20.0,   0.0],
                     [0.0,          0.0,            0.0,    20.0]])
    b_area: List[Gaussian] = []
    for i in range(1,num_h):
        for j in range(1, num_w): 
            mean = vstack([j*obj_w/(num_h+1), i*obj_h/(num_h+1), 0.0, 0.0])
            b_area.append(Gaussian(mean, cov_edge))
    
    
    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)
    birth_belief.extend(b_area)

    return birth_belief


birth_belief = phd_BirthModels(8, 6)

# phd filter

phd = GaussianMixturePHD(
        birth_belief,
        survival_rate,
        detection_rate,
        intensity,
        F,
        H,
        Q,
        R
)

# measurements
meas: List[ndarray] = []
for k in range(1,20):
    meas.insert(k,  ObjectHandler.getObjectStates(k))

meas_v: List[ndarray] = []
for k in range(len(meas)):
    meas_vk: ndarray = []

    for j in range(len(meas[k])):
        meas_vk.append(array([[meas[k][j][0]], [meas[k][j][1]]]))
        #print('----')
        #print(meas_vi)
    meas_v.insert(k, meas_vk)

# run filter
pos_phd: List[ndarray] = []
for z in meas_v:
    phd.predict()
    phd.correct(z)
    #pruning
    phd.prune(array([0.01]), array([50]), 100)
    pos_phd.append(phd.extract())
    
    print('--------------')
    print('extract data:')
    print(phd.extract())

# plot
for i in range(19):
    #Messungen
    for j in range(len(meas_v[i])):
        plt.plot(meas_v[i][j][0],meas_v[i][j][1],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        plt.plot(pos_phd[i][l][0],pos_phd[i][l][1],'ro',color= 'red', ms= 3)
        
#plt.legend(['Zk', 'phd'])     
plt.title('x-y-Raum')
plt.xlabel('x-Koord.')
plt.ylabel('y-Koord.')
plt.axis([-5,645,-5,485])
plt.gca().invert_yaxis()
plt.show()