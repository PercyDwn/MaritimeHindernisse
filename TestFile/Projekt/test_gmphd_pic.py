 ############test##################
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
# Typing
from typing import List

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('C:/Users/lukas/source/repos/PercyDwn/MaritimeHindernisse/TestFile/Projekt/Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)

for i in range(1,20):
    print('---------------------------------------')
    success = ObjectHandler.updateObjectStates()
    if success == True:
        print('updated states for time step ' + str(i))
    else:
        print('could not update states for time step ' + str(i))
    #print('last object states:')
    #print(ObjectHandler.getLastObjectStates())
    #cv2.waitKey(1000)

print('---------------------------------------')
#print('current time step: ' + str(ObjectHandler.getTimeStepCount()))
#print(ObjectHandler.getLastObjectStates())
#print('---------------------------------------')

for i in range(1,20):
    try:
        print('get data for timestep ' + str(i) + ':')
    
        print(ObjectHandler.getObjectStates(i))
    except InvalidTimeStepError as e:
        print(e.args[0])
    print('---------------------------------------')

# Messungen
#------------------------------------------------------------------------
meas: List[ndarray] = []
for i in range(1,20):
    meas.insert(i,  ObjectHandler.getObjectStates(i))

meas_v: List[ndarray] = []
for i in range(len(meas)):
    meas_vi: ndarray = []

    for j in range(len(meas[i])):
        meas_vi.append(array([[meas[i][j][0]], [meas[i][j][1]]]))
        print('----')
        print(meas_vi)
    meas_v.insert(i, meas_vi)


#GM_PHD filter initialisieren
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
# x= [x, y, dx, dy]

F = array([[1.0, 0.0, 1.0, 0.0], 
           [0.0, 1.0, 0.0, 1.0], 
           [0.0, 0.0, 1.0, 0.0], 
           [0.0, 0.0, 0.0, 1.0]])
H = array([[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0]])
Q = 1*eye(4)
R = 0.1*eye(2)

#Def. Birth_belief
mean1 = vstack([0.0, 120.0, 10.0, 7.0])
covariance1 = array([[30, 0.0, 0.0, 0.0], 
                     [0.0, 5.0, 0.0, 0.0],
                     [0.0, 0.0, 10.0, 0.0],
                     [0.0, 0.0, 0.0, 10.0]])

mean2 = vstack([600.0, 120.0, -1.0, 1.0])
covariance2 = array([[5, 0.0, 0.0, 0.0], 
                     [0.0, 30.0, 0.0, 0.0],
                     [0.0, 0.0, 2.0, 0.0],
                     [0.0, 0.0, 0.0, 2.0]])
birth_belief = [Gaussian(mean1, covariance1), Gaussian(mean2, covariance2)]

survival_rate = 0.9999
detection_rate = 0.9
intensity = 0.01
p_d = 0.95 #Detektionsrate

phd = GaussianMixturePHD(
                birth_belief,
                survival_rate,
                p_d,
                intensity,
                F,
                H,
                Q,
                R)



# PHD-Filter auf DATA anwenden
#------------------------------------------------------------------------
pos_phd: List[ndarray] = []
for z in meas_v:
    phd.predict()
    phd.correct(z)
    pos_phd.append(phd.extract())
    print(phd.extract())
    print('--------------')
    #pruning
    phd.prune(array([0.3]), array([3]), 10)


# Plott DATA
#------------------------------------------------------------------------

for i in range(19):
    #Messungen
    for j in range(len(meas_v[i])):
        plt.plot(meas_v[i][j][0],meas_v[i][j][1],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
        plt.plot(pos_phd[i][l][0],pos_phd[i][l][1],'ro',color= 'red', ms= 3)
        
plt.legend(['Zk', 'phd'])     
plt.title('x-y-Raum')
plt.xlabel('x-Koord.')
plt.ylabel('y-Koord.')
plt.axis([-5,650,-5,650])
plt.show()





