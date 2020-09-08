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
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Typing
from typing import List, Tuple

ObjectHandler = ObjectHandler()
#Dateipfad individuell anpassen!!
#################################
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

for i in range(1,20):
    try:
        print('get data for timestep ' + str(i) + ':')
    
        print(ObjectHandler.getObjectStates(i))
    except InvalidTimeStepError as e:
        print(e.args[0])
    print('---------------------------------------')

#------------------------------------------------------------------------
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
        #print('----')
        #print(meas_vi)
    meas_v.insert(i, meas_vi)

#------------------------------------------------------------------------
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
Q = .1*eye(4)
R = 0.1*eye(2)

#Def. Birth_belief
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

survival_rate = 0.99
detection_rate = 0.9
intensity = 0.01

phd = GaussianMixturePHD(
                birth_belief,
                survival_rate,
                detection_rate,
                intensity,
                F,
                H,
                Q,
                R)


#------------------------------------------------------------------------
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

#------------------------------------------------------------------------
# Plott DATA
#------------------------------------------------------------------------
#------------------------------------------------------------------------

#x-y Raum
#------------------------------------------------------------------------
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
plt.axis([-5,650,-5,650])
plt.show()

# x Achse
#------------------------------------------------------------------------
K = np.arange(len(meas_v))

for i in K:
    #Messungen
    for j in range(len(meas_v[i])):
        plt.plot(meas_v[i][j][0],K[i],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        plt.plot(pos_phd[i][l][0],K[i],'ro',color= 'red', ms= 3)
        
#plt.legend(['Zk', 'phd'])     
plt.title('x-Raum')
plt.xlabel('x-Koord.')
plt.ylabel('zeitpunkt k')
plt.axis([-5,650,-5,20])
plt.show()

# y-Achse
#------------------------------------------------------------------------
for i in K:
    #Messungen
    for j in range(len(meas_v[i])):
        plt.plot(meas_v[i][j][1],K[i],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        plt.plot(pos_phd[i][l][1],K[i],'ro',color= 'red', ms= 3)
        
#plt.legend(['Zk', 'phd'])     
plt.title('y-Raum')
plt.xlabel('y-Koord.')
plt.ylabel('zeitpunkt k')
plt.axis([-5,650,-5,20])
plt.show()

# Scater plot 3D
#------------------------------------------------------------------------
#fig = plt.figure()
#ax = Axes3D(fig)

#for i in K:
#    for j in range(len(meas_v[i])):
#        ax.scatter(meas_v[i][j][0],meas_v[i][j][1],K[i])

#ax.set_xlabel('X Axis')
#ax.set_ylabel('Y Axis')
#ax.set_zlabel('k')
#plt.show()


#------------------------------------------------------------------------
# Plott DATA in Image
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#for i in range(1,20):
#    ObjectHandler.updateObjectStates(True)
#    cv2.drawMarker(img, obst.bottom_center, (0, 255, 0), cv2.MARKER_CROSS, 10, thickness=2)