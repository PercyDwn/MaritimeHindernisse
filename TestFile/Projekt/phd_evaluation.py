
from ObjectHandler import *

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
from phd_functions import * 
from boatMove import *

# Typing
from typing import List, Tuple, cast

# GM_PHD filter init
F = array([[1.0, 0.0, 1.0, 0.0], 
           [0.0, 1.0, 0.0, 1.0], 
           [0.0, 0.0, 1.0, 0.0], 
           [0.0, 0.0, 0.0, 1.0]])
H = array([[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0]])
Q = 20*eye(4)
R = array(
    [[7, 0],
     [0, 50]])
Q = .1*eye(4)
R = .05*eye(2)

# image size
img_w = 50
img_h = 50

birth_belief = phd_BirthModels(obj_w = img_w, obj_h = img_h, num_w = 5, num_h = 5, ypt = 0, xpe = 0)
# birth_belief = phd_BirthModelsOld(num_w = 10, num_h = 10)

fig = plotGMM(gmm = birth_belief, pixel_w = img_w, pixel_h = img_h, figureTitle = 'Birth Belief GMM', savePath = '/PHD_Plots')
# fig.show()

# phd settings
survival_rate = 0.999
detection_rate = 0.9
intensity = 0.05

# phd object
phd = GaussianMixturePHD(birth_belief, survival_rate, detection_rate, intensity, F, H, Q, R)


# generate measurements
print('generate measurements ...')
meas: List[ndarray] = []

obj1_start = (5,5)
obj1_end = (45,45)

obj2_start = (5,45)
obj2_end = (15,5)

obj3_start = (35,25)
obj3_end = (30,15)

obj1_pos = BoatPos(obj1_start,obj1_end,1,25)
obj2_pos = BoatPos(obj2_start,obj2_end,1,25)
obj3_pos = BoatPos(obj3_start,obj3_end,1,25)

plt.scatter(*zip(*obj1_pos),c='#000000')
#plt.scatter(*zip(*obj2_pos),c='#000000')
#plt.scatter(*zip(*obj3_pos),c='#000000')
# plt.show()

for k in range(0,25):
    # meas.insert(k, [obj1_pos[k],obj2_pos[k],obj3_pos[k]])
    meas.insert(k, [obj1_pos[k]])

print('rearrange measurements ...')
meas_v: List[ndarray] = []
for k in range(len(meas)):
    meas_vk: ndarray = []

    for j in range(len(meas[k])):
        meas_vk.append(array([[meas[k][j][0]], [meas[k][j][1]]]))
    meas_v.insert(k, meas_vk)

# apply phd filter
print('start phd filtering...')
pos_phd: List[ndarray] = []
ci = 1
plt.figure(1)
for z in meas_v:
    phd.predict()
    phd.correct(z)
    # fig = plotGMM(gmm = phd.gmm, pixel_w = 640, pixel_h = 480, detail = 1 , method = 'rowwise', figureTitle = 'PHD GMM k-' + str(ci)+'before pruning', savePath = '/PHD_Plots')
    phd.prune(array([0.01]), array([20]), 100)
    pos_phd.append(phd.extract())
    print( 'timestep ' + str(ci) )
    print( 'tracking ' + str(len(phd.extract())) + ' objects' )
    fig = plotGMM(gmm = phd.gmm, pixel_w = 50, pixel_h = 50, detail = 1 , method = 'rowwise', figureTitle = 'PHD GMM k-' + str(ci)+'after pruning', savePath = '/PHD_Plots')
    for l in range(len(z)):
       plt.plot(z[l][0],z[l][1],'ro',color= 'black', ms= 1)
    for est in phd.extract():
       plt.plot(est[0],est[1],'ro',color= 'red', ms= 1)
    #plt.show()
    fig.close()
    ci += 1
    print('------------------')

# xy 
for i in range(len(meas_v)):
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
plt.gca().invert_yaxis()
plt.show()