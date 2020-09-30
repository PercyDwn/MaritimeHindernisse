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

def gauss(x, mean, cov, normalized=True):
    if len(x.shape) == 2:
        mean = mean[:, None]
    value = np.exp(-.5 * np.sum(np.sum((x[None, :] - mean[None, :]) * np.linalg.inv(cov)[:, :, None], axis=1) * (x - mean), axis=0))
    if normalized:
        value *= 1/2./pi/np.sqrt(np.linalg.det(cov))
    return value

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)

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
Q = 20*eye(4)
R = array(
  [[7, 0],
   [0, 50]]
)

mean1 = vstack([1.0, 120.0, 10.0, 2.0])
covariance1 = array([[10, 0.0, 0.0, 0.0], 
                     [0.0, 5.0, 0.0, 0.0],
                     [0.0, 0.0, 10.0, 0.0],
                     [0.0, 0.0, 0.0, 10.0]])

mean2 = vstack([1.0, 120.0, 17.0, 2.0])
covariance2 = array([[15, 0.0, 0.0, 0.0], 
                     [0.0, 30.0, 0.0, 0.0],
                     [0.0, 0.0, 5.0, 0.0],
                     [0.0, 0.0, 0.0, 2.0]])
birth_belief = [Gaussian(mean1, covariance1), Gaussian(mean2, covariance2)]

obj_h = 480
obj_w = 640

def phd_BirthModels (num_w: int, num_h: int) -> List[Gaussian]:
    """
     Args:
            ObjectHandler: ObjectHandler          
            num_w: number of fields on width
            num_h: number of Fields on height
            
    """

     #ObjectHandler updaten
    #--------------------------
    k = 1   #Zeitschritt
    pictures_availiable = True
    fig = plt.figure()
    est_phd: ndarray = []

    """ while pictures_availiable == True: #While: 
        try:
            #ObjectHandler.updateObjectStates()
            current_measurement_k = ObjectHandler.getObjectStates(k) #Daten der Detektion eines Zeitschrittes 
            #print(current_measurement_k)
            #print([current_measurement_k[0][0]])
        except InvalidTimeStepError as e:
            print(e.args[0])
            k = 0                
            pictures_availiable = False
            break
        k = k+1
    # Bild höhe und breite Abrufen
    obj_h = ObjectHandler.getImgHeight()
    obj_w = ObjectHandler.getImgWidth() """

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
            mean = vstack([j*obj_w/(num_w+1), i*obj_h/(num_h+1), 0.0, 0.0])
            b_area.append(Gaussian(mean, cov_edge))
    
    
    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)
    birth_belief.extend(b_area)

    return birth_belief


birth_belief = phd_BirthModels(8, 6)

survival_rate = 0.999
detection_rate = 0.9
intensity = 0.0001

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

meas: List[ndarray] = []
meas_v: List[ndarray] = []
pos_phd: List[ndarray] = []
ObjectHandler.setPlotOnUpdate(True)

inspect = 5
detail = 5

for k in range(1,inspect+1):
    meas.insert(k,  ObjectHandler.getObjectStates(k, 'cc'))
    meas_vk: ndarray = []
    for j in range(len(meas[k-1])):
        meas_vk.append(array([[meas[k-1][j][0]], [meas[k-1][j][1]]]))
    meas_v.insert(k-1, meas_vk)

    z = meas_v[k-1]
    phd.predict()
    phd.correct(z)
    phd.prune(array([0.01]), array([50]), 100)
    pos_phd.append(phd.extract())
    #print(phd.extract())
    #print('--------------')

    if k == inspect:

      plt = plotGMM(phd.gmm, pixel_w = obj_w, pixel_h = obj_h, detail = detail)
      """ for l in range(len(pos_phd[k-1])):
        plt.plot(pos_phd[k-1][l][0],pos_phd[k-1][l][1],'ro',color= 'white', ms= 3) """
      
      plt.show()
      # print("heightmap printed")
      cv2.waitKey(0)



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