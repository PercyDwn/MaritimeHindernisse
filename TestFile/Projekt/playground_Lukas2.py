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
from typing import List, Tuple
from plotGMM import *


#def gauss(x, mean, cov, normalized=True):
#    if len(x.shape) == 2:
#        mean = mean[:, None]
#    value = np.exp(-.5 * np.sum(np.sum((x[None, :] - mean[None, :]) * np.linalg.inv(cov)[:, :, None], axis=1) * (x - mean), axis=0))
#    if normalized:
#        value *= 1/2./pi/np.sqrt(np.linalg.det(cov))
#    return value

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)

obj_h = 480
obj_w = 640

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
    cov_edge = array([[75,  0.0,             0.0, 0.0], 
                     [0.0,  (obj_h/(num_h)),   0.0, 0.0],
                     [0.0,  0.0,             100.0, 0.0],
                     [0.0,  0.0,             0.0, 100.0]])
    for i in range(num_h):
        mean = vstack([30,  i*obj_h/num_h+obj_h/(2*num_h), 10.0, 0.0])
        #print(i*obj_h/num_h+obj_h/(2*num_h))
        #print('--------------')
        b_leftside.append(Gaussian(mean, 40*cov_edge, 0.1))   
    # Birthmodelle Rand rechts
    #--------------------------
    b_rightside: List[Gaussian] = [] 
    for i in range(num_h):
        mean = vstack([obj_w-30,  i*obj_h/num_h+obj_h/(2*num_h), -10.0, 0.0])
        b_rightside.append(Gaussian(mean, 40*cov_edge, 0.1))
    # Birthmodelle übers Bild
    #--------------------------
    cov_area = array([[(obj_w/num_w), 0.0,            0.0,    0.0], 
                     [0.0,          (obj_h/(num_h)),  0.0,    0.0],
                     [0.0,          0.0,            100.0,   0.0],
                     [0.0,          0.0,            0.0,    100.0]])
    b_area: List[Gaussian] = []
    for i in range(num_h):
        for j in range(num_w): 
            mean = vstack([j*obj_w/num_w+obj_w/(2*num_w), i*obj_h/num_h+obj_h/(2*num_h), 0.0, 0.0])
            b_area.append(Gaussian(mean, 40*cov_area, 0.5))
            
    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)
    birth_belief.extend(b_area)

    return birth_belief


birth_belief = phd_BirthModels(5, 4) #12, 10

#fig = plotGMM(birth_belief, 640, 480)
#plt.title('gausplot des birthmodels')
#plt.show()
maxWeight = 0
for comp in birth_belief:
    if comp.w > maxWeight:
       maxWeight = comp.w
print('max weight: ' + str(maxWeight))
minWeight = 1
for comp in birth_belief:
    if comp.w < minWeight:
        minWeight = comp.w
print('min weight: ' + str(minWeight))

numGaus = len(birth_belief)
print('Num of Gaus in gmm: ' + str(numGaus))

survival_rate = 0.999
detection_rate = 0.9
intensity = 0.000075
T = 1
F = array(
  [[1.0, 0.0, T,    0.0], 
   [0.0, 1.0, 0.0,  T], 
   [0.0, 0.0, 1.0,  0.0], 
   [0.0, 0.0, 0.0,  1.0]]
)
H = array(
  [[1.0, 0.0, 0.0, 0.0],
   [0.0, 1.0, 0.0, 0.0]]
)
Q = 20*array([[(T**4)/4,  0,      (T**3)/2, 0],
              [0,       (T**4)/4, 0,      (T**3)/2],
              [(T**3)/2,  0,      T**2,   0], 
              [0,      (T**3)/2,  0,      T**2]
              ])
R = .001*array(
  [[1, 0],
   [0, 2]]
)

phd = GaussianMixturePHD(
  birth_belief,
  survival_rate,
  detection_rate,
  intensity,
  F,
  H,
  Q,
  R)

meas: List[ndarray] = []
meas_v: List[ndarray] = []
pos_phd: List[ndarray] = []
ObjectHandler.setPlotOnUpdate(True)

inspect = 10

for k in range(1,21):
    meas.insert(k,  ObjectHandler.getObjectStates(k, 'cb'))
    meas_vk: ndarray = []
    for j in range(len(meas[k-1])):
        meas_vk.append(array([[meas[k-1][j][0]], [meas[k-1][j][1]]]))
    meas_v.insert(k-1, meas_vk)

    z = meas_v[k-1]
    phd.predict()
    phd.correct(z)
    print('Anzahl an Gaussummanden vorm pruning: ' + str(len(phd.gmm)))
    #fig = plotGMM(phd.gmm, 640, 480)
    #for l in range(len(meas_v[k-1])):
    #        plt.plot(meas_v[k-1][l][0],meas_v[k-1][l][1],'ro',color= 'grey', ms= 1)
    #for est in phd.extract():
    #        plt.plot(est[0],est[1],'ro',color= 'red', ms= 1)
    #plt.show()
    phd.prune(array([0.01]), array([5]), 8)
    print('Anzahl an Gaussummanden nachm pruning: ' + str(len(phd.gmm)))
    fig = plotGMM(phd.gmm, 640, 480)
    for l in range(len(meas_v[k-1])):
            plt.plot(meas_v[k-1][l][0],meas_v[k-1][l][1],'ro',color= 'grey', ms= 1)
    for est in phd.extract():
            plt.plot(est[0],est[1],'ro',color= 'red', ms= 1)
    #plt.show()
    fig.close()
    pos_phd.append(phd.extract(0.5))
    #print(phd.extract())
    #print('--------------')
    maxWeight = 0
    for comp in phd.gmm:
        if comp.w > maxWeight:
            maxWeight = comp.w
    print('max weight: ' + str(maxWeight))
    minWeight = 1
    for comp in phd.gmm:
        if comp.w < minWeight:
            minWeight = comp.w
    print('min weight: ' + str(minWeight))
    if k == inspect:
        fig = plt.figure
        fig = plotGMM(phd.gmm, 640, 480)

        maxWeight = 0
        for comp in phd.gmm:
            if comp.w > maxWeight:
                maxWeight = comp.w
        print('max weight: ' + str(maxWeight))
        minWeight = 1
        for comp in phd.gmm:
            if comp.w < minWeight:
                minWeight = comp.w
        print('min weight: ' + str(minWeight))

        numGaus = len(phd.gmm)
        print('Num of Gaus in gmm: ' + str(numGaus))
           
        for l in range(len(meas_v[k-1])):
            plt.plot(meas_v[k-1][l][0],meas_v[k-1][l][1],'ro',color= 'grey', ms= 1)
        for l in range(len(pos_phd[k-1])):
            plt.plot(pos_phd[k-1][l][0],pos_phd[k-1][l][1],'ro',color= 'red', ms= 1)
        plt.show()


K = np.arange(len(meas_v))
for i in K:
    #Messungen
    for j in range(len(meas_v[i])):
        plt.plot(meas_v[i][j][0],meas_v[i][j][1],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        plt.plot(pos_phd[i][l][0],pos_phd[i][l][1],'ro',color= 'red', ms= 3)
        
plt.title('x-y-Raum')
plt.xlabel('x-Koord.')
plt.ylabel('y-Koord.')
plt.gca().invert_yaxis()
plt.show()

    


""" for i in range(19):
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
plt.show() """


#def phd_BirthModels (num_w: int, num_h: int) -> List[Gaussian]:
#    """
#     Args:
#            ObjectHandler: ObjectHandler          
#            num_w: number of fields on width
#            num_h: number of Fields on height
            
#    """


#    obj_h = 480
#    obj_w = 640

#    birth_belief: List[Gaussian] = []

#    # Birthmodelle Rand links
#    #--------------------------
#    b_leftside: List[Gaussian] = [] 
#    cov_edge = array([[120*60,  0.0,             0.0, 0.0], 
#                     [0.0,  (obj_h/(num_h)),   0.0, 0.0],
#                     [0.0,  0.0,             10.0*10, 0.0],
#                     [0.0,  0.0,             0.0, 10.0*10]])
#    for i in range(num_h):
#        mean = vstack([20,  i*obj_h/num_h+obj_h/(2*num_h), 10.0, 0.0])
#        print(i*obj_h/num_h+obj_h/(2*num_h))
#        print('--------------')
#        b_leftside.append(Gaussian(mean, cov_edge, 0.05))
    
#    # Birthmodelle Rand rechts
#    #--------------------------
#    b_rightside: List[Gaussian] = [] 
#    for i in range(num_h):
#        mean = vstack([obj_w-20,  i*obj_h/num_h+obj_h/(2*num_h), -10.0, 0.0])
#        print(i*obj_h/num_h+obj_h/(2*num_h))
#        b_rightside.append(Gaussian(mean, cov_edge, 0.05))

 
#    # Birthmodelle übers Bild
#    #--------------------------
#    cov_area = array([[(obj_w/num_w), 0.0,            0.0,    0.0], 
#                     [0.0,          (obj_h/(num_h)),  0.0,    0.0],
#                     [0.0,          0.0,            10.0*10,   0.0],
#                     [0.0,          0.0,            0.0,    10.0*10]])
#    b_area: List[Gaussian] = []
#    for i in range(num_h):
#        for j in range(num_w): 
#            mean = vstack([j*obj_w/num_w+obj_w/(2*num_w), i*obj_h/num_h+obj_h/(2*num_h), 0.0, 0.0])
#            b_area.append(Gaussian(mean, cov_area, 0.1))
    
    
#    birth_belief.extend(b_leftside)
#    birth_belief.extend(b_rightside)
#    birth_belief.extend(b_area)

#    return birth_belief


#birth_belief = phd_BirthModels(8, 6)
##fig = plt.figure
##fig = plotGMM(birth_belief, 640, 480, 5)
##plt.title('GausPlot des Birthmodels')
##plt.show()
##maxWeight = 0
##for comp in birth_belief:
##    if comp.w > maxWeight:
##       maxWeight = comp.w
##print('max weight: ' + str(maxWeight))
##minWeight = 1
##for comp in birth_belief:
##    if comp.w < minWeight:
##        minWeight = comp.w
##print('min weight: ' + str(minWeight))

##numGaus = len(birth_belief)
##print('Num of Gaus in gmm: ' + str(numGaus))

#for k in range(1,20):
#    meas.insert(k,  ObjectHandler.getObjectStates(k, 'cc'))
#    meas_vk: ndarray = []
#    for j in range(len(meas[k-1])):
#        meas_vk.append(array([[meas[k-1][j][0]], [meas[k-1][j][1]]]))
#    meas_v.insert(k-1, meas_vk)

#    z = meas_v[k-1]
#    phd.predict()
#    phd.correct(z)
#    phd.prune(array([0.0001]), array([3]), 20)
#    pos_phd.append(phd.extract(0.01))
#    #print(phd.extract())
#    #print('--------------')
#    if k == inspect:
#        fig = plt.figure
#        fig = plotGMM(phd.gmm, 640, 480, 5)

#        maxWeight = 0
#        for comp in phd.gmm:
#            if comp.w > maxWeight:
#                maxWeight = comp.w
#        print('max weight: ' + str(maxWeight))
#        minWeight = 1
#        for comp in phd.gmm:
#            if comp.w < minWeight:
#                minWeight = comp.w
#        print('min weight: ' + str(minWeight))

#        numGaus = len(phd.gmm)
#        print('Num of Gaus in gmm: ' + str(numGaus))
           
#        for l in range(len(meas_v[k-1])):
#            plt.plot(meas_v[k-1][l][0],meas_v[k-1][l][1],'ro',color= 'white', ms= 3)
#        for l in range(len(pos_phd[k-1])):
#            plt.plot(pos_phd[k-1][l][0],pos_phd[k-1][l][1],'ro',color= 'red', ms= 3)
#        plt.show()


#K = np.arange(len(meas_v))
#for i in K:
#    #Messungen
#    for j in range(len(meas_v[i])):
#        plt.plot(meas_v[i][j][0],meas_v[i][j][1],'ro',color='black')

#    #Schätzungen
#    for l in range(len(pos_phd[i])):
#        plt.plot(pos_phd[i][l][0],pos_phd[i][l][1],'ro',color= 'red', ms= 3)
        
##plt.legend(['Zk', 'phd'])     
#plt.title('x-y-Raum')
#plt.xlabel('x-Koord.')
#plt.ylabel('y-Koord.')
#plt.gca().invert_yaxis()
#plt.show()
