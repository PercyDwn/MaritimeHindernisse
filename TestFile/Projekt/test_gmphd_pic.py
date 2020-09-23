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
from mpl_toolkits.mplot3d import Axes3D
import cv2

from phd_plot import *

# Typing
from typing import List, Tuple

#------------------------------------------------------------------------
# ObjectHandler initialisieren
#------------------------------------------------------------------------
ObjectHandler = ObjectHandler()
#Dateipfad individuell anpassen!!
#################################
ObjectHandler.setImageFolder('C:/Users/lukas/source/repos/PercyDwn/MaritimeHindernisse/TestFile/Projekt/Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)

#------------------------------------------------------------------------
# GM_PHD filter initialisieren
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
Q = 20*eye(4)
#R = 30*eye(2)
R = array([[7, 0],
           [0, 50]])

#Def. Birth_belief

#birth_belief = phd_BirthModels(ObjectHandler, 4, 3)

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

    while pictures_availiable == True: #While: 
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
    obj_w = ObjectHandler.getImgWidth()

    #obj_h = 480
    #obj_w = 640

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

#mean = vstack([320, 240, 0.0, .0])
#covariance = array([[1000, 0.0, 0.0, 0.0], 
#                    [0.0, 1000.0, 0.0, 0.0],
#                    [0.0, 0.0, 50.0, 0.0],
#                    [0.0, 0.0, 0.0, 50.0]])
#birth_belief = [Gaussian(mean, covariance)]

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
                R)





def gm_phd(phd, ObjectHandler) -> ndarray:
    k = 1   #Zeitschritt
    pictures_availiable = True
    fig = plt.figure()
    est_phd: ndarray = []

    while pictures_availiable == True: #While: 
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
        #current_measurement_k = ObjectHandler.getObjectStates(k+1) #Daten der Detektion eines Zeitschrittes 
        meas_vk: ndarray = []
        for j in range(len(current_measurement_k)):
            meas_vk.append(array([[current_measurement_k[j][0]], [current_measurement_k[j][1]]]))
        #meas_v.insert(k, meas_vk)
        phd.predict()
        phd.correct(meas_vk)
        est_phd.append(phd.extract())
        #print(phd.extract())
        #print('--------------')
        #pruning
        phd.prune(array([0.001]), array([35]), 120)
        plot_PHD_realpic(ObjectHandler, est_phd, meas_vk, k)
        k = k+1
            
    plt.grid()
    #plt.show()   
    return est_phd 

#------------------------------------------------------------------------
#pos_phd = gm_phd(phd, ObjectHandler)
#------------------------------------------------------------------------





#------------------------------------------------------------------------
# Messungen
#------------------------------------------------------------------------
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




#------------------------------------------------------------------------
# PHD-Filter auf DATA anwenden
#------------------------------------------------------------------------
pos_phd: List[ndarray] = []


for z in meas_v:
    phd.predict()
    phd.correct(z)
    #pruning
    phd.prune(array([0.01]), array([50]), 100)
    pos_phd.append(phd.extract())
    print(phd.extract())
    print('--------------')
    

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
plt.axis([-5,645,-5,485])
plt.gca().invert_yaxis()
plt.show()

# x Achse
#------------------------------------------------------------------------
K = np.arange(len(meas_v))

for i in K:
    #Messungen
    for j in range(len(meas_v[i])):
        plt.plot(K[i], meas_v[i][j][0],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        plt.plot(K[i], pos_phd[i][l][0],'ro',color= 'red', ms= 3)
        
#plt.legend(['Zk', 'phd'])     
plt.title('x-Raum')
plt.xlabel('zeitpunkt k')
plt.ylabel('x-Koord.')
plt.axis([-1,20,-5,645])
plt.show()

# y-Achse
#------------------------------------------------------------------------
for k in K:
    #Messungen
    for j in range(len(meas_v[k])):
        plt.plot(K[k], meas_v[k][j][1],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[k])):
        plt.plot(K[k], pos_phd[k][l][1],'ro',color= 'red', ms= 3)
        
#plt.legend(['Zk', 'phd'])     
plt.title('y-Raum')
plt.xlabel('zeitpunkt k')
plt.ylabel('y-Koord.')
plt.axis([-1,20,-5,485])
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
#f()


#------------------------------------------------------------------------
# Plott DATA in Image
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#for i in range(1,20):
#    ObjectHandler.updateObjectStates(True)
#    cv2.drawMarker(img, obst.bottom_center, (0, 255, 0), cv2.MARKER_CROSS, 10, thickness=2)

#for k in range(1,20):
#    plot_PHD_realpic(ObjectHandler, pos_phd, k)



#number_states = len(F) # Zuständezahl
#k = 0   #Zeitschritt
#pictures_availiable = True
#fig = plt.figure()

#for i in range(1,20):
#    print('---------------------------------------')
#    success = ObjectHandler.updateObjectStates()
#    if success == True:
#        print('updated states for time step ' + str(i))
#    else:
#        print('could not update states for time step ' + str(i))
#    print('last object states:')
#    print(ObjectHandler.getLastObjectStates())
#    #cv2.waitKey(1000)

#print('---------------------------------------')
##print('current time step: ' + str(ObjectHandler.getTimeStepCount()))
##print(ObjectHandler.getLastObjectStates())

#for i in range(1,20):
#    try:
#        print('get data for timestep ' + str(i) + ':')
    
#        print(ObjectHandler.getObjectStates(i))
#    except InvalidTimeStepError as e:
#        print(e.args[0])
#    print('---------------------------------------')