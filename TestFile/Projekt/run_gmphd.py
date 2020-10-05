# import stuff

from ObjectHandler import ObjectHandler

import matplotlib.pyplot as plt

import numpy as np
from numpy import vstack, array, ndarray, eye

import math 

from mpl_toolkits.mplot3d import Axes3D

import cv2

from typing import List, Tuple

# init ObjectHandler
ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('Bilder/list1')
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

mean = vstack([320, 240, 0.0, .0])
covariance = array(
    [[1000, 0.0, 0.0, 0.0], 
    [0.0, 1000.0, 0.0, 0.0],
    [0.0, 0.0, 50.0, 0.0],
    [0.0, 0.0, 0.0, 50.0]]
)
birth_belief = [Gaussian(mean, covariance)]

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
for z in meas_v:
    phd.predict()
    phd.correct(z)
    #pruning
    phd.prune(array([0.01]), array([50]), 100)
    pos_phd.append(phd.extract())
    print(phd.extract())
    print('--------------')

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