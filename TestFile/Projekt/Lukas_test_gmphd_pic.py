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
from plotGMM import *

# Typing
from typing import List, Tuple

#------------------------------------------------------------------------
# ObjectHandler initialisieren
#------------------------------------------------------------------------
ObjectHandler = ObjectHandler()
#Dateipfad individuell anpassen!!
#################################
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1')
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
Q = 10*eye(4)
#R = 30*eye(2)
R = array([[5, 0],
           [0, 25]])

#Def. Birth_belief
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
    cov_edge = array([[3000,  0.0,             0.0, 0.0], 
                     [0.0,  (obj_h/(num_h))*40,   0.0, 0.0],
                     [0.0,  0.0,             50.0*5, 0.0],
                     [0.0,  0.0,             0.0, 50.0*5]])
    for i in range(num_h):
        mean = vstack([20,  i*obj_h/num_h+obj_h/(2*num_h), 10.0, 0.0])
        b_leftside.append(Gaussian(mean, cov_edge, 0.05))
    
    # Birthmodelle Rand rechts
    #--------------------------
    b_rightside: List[Gaussian] = [] 
    for i in range(num_h):
        mean = vstack([obj_w-20,  i*obj_h/num_h+obj_h/(2*num_h), -10.0, 0.0])
        b_rightside.append(Gaussian(mean, cov_edge, 0.05))

 
    # Birthmodelle übers Bild
    #--------------------------
    cov_area = array([[(obj_w/num_w)*40, 0.0,            0.0,    0.0], 
                     [0.0,          (obj_h/(num_h))*40,  0.0,    0.0],
                     [0.0,          0.0,            50.0*5,   0.0],
                     [0.0,          0.0,            0.0,    50.0*5]])
    b_area: List[Gaussian] = []
    for i in range(num_h):
        for j in range(num_w): 
            mean = vstack([j*obj_w/num_w+obj_w/(2*num_w), i*obj_h/num_h+obj_h/(2*num_h), 0.0, 0.0])
            b_area.append(Gaussian(mean, cov_area, 0.9))
    
    
    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)
    birth_belief.extend(b_area)

    return birth_belief



birth_belief = phd_BirthModels(5, 4)

fig = plt.figure
fig = plotGMM(birth_belief, 640, 480)
plt.title('GausPlot des Birthmodels')
plt.show()

survival_rate = 0.999
detection_rate = 0.9
intensity = 0.00015

phd = GaussianMixturePHD(
                birth_belief,
                survival_rate,
                detection_rate,
                intensity,
                F,
                H,
                Q,
                R)


inspect = 15


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
        #if k == inspect:
        #    fig = plt.figure
        #    fig = plotGMM(phd.gmm, 640, 480, 3)
        #    plt.title('GausPlot befor pruning')
        #    plt.show()
        #print(phd.extract())
        #print('--------------')
        print('Anzahl an Gaussummanden vorm pruning: ' + str(len(phd.gmm)))
        #pruning
        phd.prune(array([0.001]), array([3]), 25)
        print('Anzahl an Gaussummanden nachm pruning: ' + str(len(phd.gmm)))

        est_phd.append(phd.extract())

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
            fig = plotGMM(phd.gmm, 640, 480, 4)
            for l in range(len(meas_vk)):
                plt.plot(meas_vk[l][0],meas_vk[l][1],'ro',color= 'black', ms= 1)
            for est in phd.extract():
                plt.plot(est[0],est[1],'ro',color= 'red', ms= 1)
            plt.title('GausPlot after pruning')
            plt.show()
        #plot_PHD_realpic(ObjectHandler, est_phd, meas_vk, k)
        k = k+1
            
    plt.grid()
    plt.show()   
    return est_phd 

#------------------------------------------------------------------------
pos_phd = gm_phd(phd, ObjectHandler)
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
#pos_phd: List[ndarray] = []


#for z in meas_v:
#    phd.predict()
#    phd.correct(z)
#    #pruning
#    phd.prune(array([0.001]), array([3]), 20)
#    pos_phd.append(phd.extract())
#    print(phd.extract())
#    print('--------------')
    

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

## x Achse
##------------------------------------------------------------------------
#K = np.arange(len(meas_v))

#for i in K:
#    #Messungen
#    for j in range(len(meas_v[i])):
#        plt.plot(K[i], meas_v[i][j][0],'ro',color='black')

#    #Schätzungen
#    for l in range(len(pos_phd[i])):
#        plt.plot(K[i], pos_phd[i][l][0],'ro',color= 'red', ms= 3)
        
##plt.legend(['Zk', 'phd'])     
#plt.title('x-Raum')
#plt.xlabel('zeitpunkt k')
#plt.ylabel('x-Koord.')
#plt.axis([-1,20,-5,645])
#plt.show()

## y-Achse
##------------------------------------------------------------------------
#for k in K:
#    #Messungen
#    for j in range(len(meas_v[k])):
#        plt.plot(K[k], meas_v[k][j][1],'ro',color='black')

#    #Schätzungen
#    for l in range(len(pos_phd[k])):
#        plt.plot(K[k], pos_phd[k][l][1],'ro',color= 'red', ms= 3)
        
##plt.legend(['Zk', 'phd'])     
#plt.title('y-Raum')
#plt.xlabel('zeitpunkt k')
#plt.ylabel('y-Koord.')
#plt.axis([-1,20,-5,485])
#plt.show()

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

def varPruneThresh_phd(ini: int, num: int, meas, plot: bool = True):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)

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

    birth_belief = phd_BirthModels(8, 6)
    for m in range (1,num+1):   

        

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
        pos_phd: List[ndarray] = []
        #PHD-Filter anwenden
        #-----------------------------
        for z in meas:
            phd.predict()
            phd.correct(z)
            #pruning
            phd.prune(array([m*ini]), array([3]), 20)
            pos_phd.append(phd.extract())
            #print(phd.extract())
            #print('--------------')

        pos_phd_all[m-1] = pos_phd
        #print(pos_phd_all[i-1])

    if plot:
        K = np.arange(len(meas))
        plt.gca().invert_yaxis()

        # x-y-Raum
        #------------------------------------
        for m in range(1, len(pos_phd_all)+1):
            ax = plt.subplot(num, 1, m)
            plt.gca().invert_yaxis()
            for i in K:
            #Schätzungen
  
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                
             

                for l in range(len(pos_phd_all[m-1][i])):
                    #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')   
                    plt.plot(pos_phd_all[m-1][i][l][0],pos_phd_all[m-1][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Prunethreshold is: '+str(m*ini))
                    

        plt.suptitle('x-y-Raum for Variable Prunethreshold')
        plt.show()

    return pos_phd_all

def varPruneNum_phd(ini: int, num: int, meas, plot: bool = True):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)

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
    birth_belief = phd_BirthModels(8, 6)
    for m in range (1,num+1):   

        

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
        pos_phd: List[ndarray] = []
        #PHD-Filter anwenden
        #-----------------------------
        for z in meas:
            phd.predict()
            phd.correct(z)
            #pruning
            phd.prune(array([0.001]), array([3]), m*ini)
            pos_phd.append(phd.extract())
            #print(phd.extract())
            #print('--------------')

        pos_phd_all[m-1] = pos_phd
        #print(pos_phd_all[i-1])

    if plot:
        K = np.arange(len(meas))
        

        # x-y-Raum
        #------------------------------------
        for m in range(1, len(pos_phd_all)+1):
            ax = plt.subplot(num, 1, m)
            plt.gca().invert_yaxis()
            for i in K:
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                #Schätzungen
                for l in range(len(pos_phd_all[m-1][i])):
                    #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')   
                    plt.plot(pos_phd_all[m-1][i][l][0],pos_phd_all[m-1][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Max Prune Number is: '+str(m*ini))

        plt.suptitle('x-y-Raum for Variable Max Prune Number')
        plt.show()

    return pos_phd_all

def varPrune_phd(iniThresh, ininum: int, num: int, meas, plot: bool = True):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)

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
    birth_belief = phd_BirthModels(8, 6)
    for m in range (1,num+1):   

        

        survival_rate = 0.999
        detection_rate = 0.9
        intensity = 0.001

        phd = GaussianMixturePHD(
                        birth_belief,
                        survival_rate,
                        detection_rate,
                        intensity,
                        F,
                        H,
                        Q,
                        R)
        pos_phd: List[ndarray] = []
        #PHD-Filter anwenden
        #-----------------------------
        for z in meas:
            phd.predict()
            phd.correct(z)
            #pruning
            phd.prune(array([iniThresh/m]), array([3]), m*ininum)
            pos_phd.append(phd.extract())
            #print(phd.extract())
            #print('--------------')

        pos_phd_all[m-1] = pos_phd
        #print(pos_phd_all[i-1])

    if plot:
        K = np.arange(len(meas))
        

        # x-y-Raum
        #------------------------------------
        for m in range(1, len(pos_phd_all)+1):
            ax = plt.subplot(num, 1, m)
            plt.gca().invert_yaxis()
            for i in K:
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                #Schätzungen
                for l in range(len(pos_phd_all[m-1][i])): 
                    plt.plot(pos_phd_all[m-1][i][l][0],pos_phd_all[m-1][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Max Prune Number is: '+str(ininum*m)+'Threshold is: '+ str(iniThresh/m))

        plt.suptitle('x-y-Raum for Variable Max Prune Number and Threshold')
        plt.show()

    return pos_phd_all

def varIntensity_phd(ini, num: int, meas, plot: bool = True):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)
    F = array([[1.0, 0.0, 1.0, 0.0], 
               [0.0, 1.0, 0.0, 1.0], 
               [0.0, 0.0, 1.0, 0.0], 
               [0.0, 0.0, 0.0, 1.0]])
    H = array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
    Q = 35*eye(4)
    R = array([[10, 0],
               [0, 50]])
    birth_belief = phd_BirthModels(8, 6)

    for m in range (1,num+1):      
        survival_rate = 0.99
        detection_rate = 0.9
        intensity = m*ini

        phd = GaussianMixturePHD(
                        birth_belief,
                        survival_rate,
                        detection_rate,
                        intensity,
                        F,
                        H,
                        Q,
                        R)
        pos_phd: List[ndarray] = []
        #PHD-Filter anwenden
        #-----------------------------
        for z in meas:
            phd.predict()
            phd.correct(z)
            #pruning
            phd.prune(array([0.001]), array([3]), 20)
            pos_phd.append(phd.extract(0.01))
            #print(phd.extract())
            #print('--------------')

        pos_phd_all[m-1] = pos_phd
    

    if plot:
        K = np.arange(len(meas))
        plt.gca().invert_yaxis()
        # x-y-Raum
        #------------------------------------
        for m in range(1, len(pos_phd_all)+1):
            ax = plt.subplot(num, 1, m)
            plt.gca().invert_yaxis()
            for i in K:
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                for l in range(len(pos_phd_all[m-1][i])):  
                    plt.plot(pos_phd_all[m-1][i][l][0],pos_phd_all[m-1][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Clutter intensity is: '+str(m*ini))
        plt.suptitle('x-y-Raum für Variation der Clutter intensity')
        plt.show()

    return pos_phd_all

def varQ_phd(ini, num: int, meas: List[ndarray], plot: bool = False):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)

    F = array([[1.0, 0.0, 1.0, 0.0], 
               [0.0, 1.0, 0.0, 1.0], 
               [0.0, 0.0, 1.0, 0.0], 
               [0.0, 0.0, 0.0, 1.0]])
    H = array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
    birth_belief = phd_BirthModels(8, 6)
    for m in range (1,num+1):   
        Q = m*ini*eye(4)
        R = array([[7, 0],
               [0, 50]])

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
        pos_phd: List[ndarray] = []
        #PHD-Filter anwenden
        #-----------------------------
        for z in meas:
            phd.predict()
            phd.correct(z)
            #pruning
            phd.prune(array([0.001]), array([3]), 20)
            pos_phd.append(phd.extract())
            #print(phd.extract())
            #print('--------------')

        pos_phd_all[m-1] = [pos_phd]

    
    if plot:
        K = np.arange(len(meas))
        
        # x-y-Raum
        #------------------------------------
        
        for m in range(1, num+1):
            ax = plt.subplot(num, 1, m)
            plt.gca().invert_yaxis()
            #plt.subplot(num, 1, m)
            for i in K:
            #Schätzungen
                #print('range(len(pos_phd_all[m-1][i])): ' + str(range(len(pos_phd_all[m-1][i]))))
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                for l in range(len(pos_phd_all[m-1][0][i])):                    
                    plt.plot(pos_phd_all[m-1][0][i][l][0],pos_phd_all[m-1][0][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Covarianz in Q: '+str(m*ini))
        plt.suptitle('x-y-Raum für Variation der Covarianzmatirzen Q')
        plt.show()

    return pos_phd_all


def varR_phd(ini, num: int, meas: List[ndarray], plot: bool = False):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)

    F = array([[1.0, 0.0, 1.0, 0.0], 
               [0.0, 1.0, 0.0, 1.0], 
               [0.0, 0.0, 1.0, 0.0], 
               [0.0, 0.0, 0.0, 1.0]])
    H = array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
    birth_belief = phd_BirthModels(8, 6)
    for m in range (1,num+1):   
        Q = 35*eye(4)
        R = array([[7, 0],
                   [0, ini*num]])

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
        pos_phd: List[ndarray] = []
        #PHD-Filter anwenden
        #-----------------------------
        for z in meas:
            phd.predict()
            phd.correct(z)
            #pruning
            phd.prune(array([0.001]), array([3]), 20)
            pos_phd.append(phd.extract())
            #print(phd.extract())
            #print('--------------')

        pos_phd_all[m-1] = [pos_phd]

    
    if plot:
        K = np.arange(len(meas))
        
        # x-y-Raum
        #------------------------------------
        fig = plt.figure()
        for m in range(1, num+1):
            ax = plt.subplot(num, 1, m)
            plt.gca().invert_yaxis()
            #plt.subplot(num, 1, m)
            for i in K:
            #Schätzungen
                #print('range(len(pos_phd_all[m-1][i])): ' + str(range(len(pos_phd_all[m-1][i]))))
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                for l in range(len(pos_phd_all[m-1][0][i])):                    
                    plt.plot(pos_phd_all[m-1][0][i][l][0],pos_phd_all[m-1][0][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Covarianz in R: '+str(m*ini))
        plt.suptitle('x-y-Raum für Variation der Covarianzmatirzen  R')
        plt.show()

    return pos_phd_all

######################################################
#phd_QRvar = varPruneThresh_phd(10, 5, meas_v, True)
######################################################
#######################################################
#varPruneNum = varPruneNum_phd(5, 5, meas_v, True)
#######################################################

#varPrune = varPrune_phd(0.0005, 10, 5, meas_v, True)

#######################################################
#varIntens = varIntensity_phd(0.000001, 5, meas_v)
#######################################################

#varQ = varQ_phd(10, 5, meas_v, True)

#varR = varR_phd(5, 5, meas_v, True)
#######################################################
#varBirth = varBirthNum_phd(2, 5, meas)
#######################################################
#'bei num = 10 beste ergebnisse -> entweder weil Cov passt oder mit auftauchen übereinstimmt'


#######################################################
#varPruneTh = varPruneThresh_phd(0.05, 5, meas, objects)
#######################################################
#'Schlechtere ergebenisse wenn großer UND kleiner 0.1'
