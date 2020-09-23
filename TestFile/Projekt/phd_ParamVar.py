import  matplotlib.pyplot as plt
import numpy as np
import pyrate_sense_filters_gmphd
import random
from functions import gaussian 
from functions import createTestDataSet
from functions import get_position
from pyrate_common_math_gaussian import Gaussian
from pyrate_sense_filters_gmphd import GaussianMixturePHD
from numpy import vstack
from numpy import array
from numpy import ndarray
from numpy import eye
import math
# Typing
from typing import List

#Initialisierung
#------------------------------------------------------------------------




#GM_PHD filter erzeugen
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#2D Test
# x= [x, y, dx, dy]

F = array([[1.0, 0.0, 1.0, 0.0], 
           [0.0, 1.0, 0.0, 1.0], 
           [0.0, 0.0, 1.0, 0.0], 
           [0.0, 0.0, 0.0, 1.0]])
H = array([[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0]])
Q = .1*eye(4)
R = .05*eye(2)

#Def. Birth_belief
mean1 = vstack([0.0, 15.0, 1.0, 1.0])
covariance1 = array([[5, 0.0, 0.0, 0.0], 
                     [0.0, 30.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

mean2 = vstack([50.0, 15.0, -1.0, 1.0])
covariance2 = array([[5, 0.0, 0.0, 0.0], 
                     [0.0, 30.0, 0.0, 0.0],
                     [0.0, 0.0, 2.0, 0.0],
                     [0.0, 0.0, 0.0, 2.0]])
birth_belief = [Gaussian(mean1, covariance1), Gaussian(mean2, covariance2)]

def phd_BirthModels (num_w: int, num_h: int) -> List[Gaussian]:
    """
     Args:
            ObjectHandler: ObjectHandler          
            num_w: number of fields on width
            num_h: number of Fields on height
            
    """

    

    obj_h = 50
    obj_w = 50

    birth_belief: List[Gaussian] = []

    # Birthmodelle Rand links
    #--------------------------
    b_leftside: List[Gaussian] = [] 
    cov_edge = array([[15, 0.0,         0.0, 0.0], 
                     [0.0, obj_h/(4*num_h), 0.0, 0.0],
                     [0.0, 0.0,         5.0, 0.0],
                     [0.0, 0.0,         0.0, 5.0]])
    for i in range(1,num_h):
        mean = vstack([0, i*obj_h/(num_h+1), 1.0, 1.0])
        b_leftside.append(Gaussian(mean, cov_edge))
    
    # Birthmodelle Rand rechts
    #--------------------------
    b_rightside: List[Gaussian] = [] 
    for i in range(1,num_h):
        mean = vstack([obj_w, i*obj_h/(num_h+1), -1.0, 1.0])
        b_rightside.append(Gaussian(mean, cov_edge))

    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)

    return birth_belief

objects: List[ndarray] = []
meas: List[ndarray] = []
pos_phd: List[ndarray] = []

#Objekte erstellen
#-------------------------------
for i in range(10):
    objects.insert(i, [array([[0.+i], [10.+i]]), array([[50.-1.5*i], [15.]]), array([[50.-i], [15.+i]])] )
   

for i in range(20):
    objects.insert(10+i, [array([[10.+1.5*i], [20.+i]]), array([[35.-1.5*i], [15.]]), array([[40.-i], [26.+i]])] )
    

#Messdaten erstellen
#-------------------------------
for i in range(10):
    meas.insert(i, [array([[0.+i], [10.+i]]), array([[50.-1.5*i], [15.]]), array([[50.-i], [15.+i]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]) ] )
  

for i in range(20):
    meas.insert(10+i, [ array([[10.+1.5*i], [20.+i]]), array([[35.-1.5*i], [15.]]), array([[40.-i], [26.+i]]), array([[50*random.random()], [50*random.random()]]),  array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]) ] )
    


def varQR_phd(ini, num: int, meas: List[ndarray], objects, plot: bool = False):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)

    F = array([[1.0, 0.0, 1.0, 0.0], 
               [0.0, 1.0, 0.0, 1.0], 
               [0.0, 0.0, 1.0, 0.0], 
               [0.0, 0.0, 0.0, 1.0]])
    H = array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
    birth_belief = phd_BirthModels(10, 10)
    for m in range (1,num+1):   
        Q = 2*m*ini*eye(4)
        R = m*ini*eye(2)

        survival_rate = 0.99
        detection_rate = 0.9
        intensity = 0.05

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
            phd.prune(array([0.1]), array([3]), 20)
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
            #plt.subplot(num, 1, m)
            for i in K:
            #Schätzungen
                #print('range(len(pos_phd_all[m-1][i])): ' + str(range(len(pos_phd_all[m-1][i]))))
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                #Objekte
                for j in range(len(objects[i])):
                    plt.plot(objects[i][j][0],objects[i][j][1],'ro',color='green')

                for l in range(len(pos_phd_all[m-1][0][i])):                    
                    plt.plot(pos_phd_all[m-1][0][i][l][0],pos_phd_all[m-1][0][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Covarianz in Q: '+str(2*m*ini)+' and R is: '+str(m*ini))
        plt.suptitle('x-y-Raum für Variation der Covarianzmatirzen Q und R')
        plt.axis([0,50,0,50])
        plt.show()

    return pos_phd_all




def varIntensity_phd(ini, num: int, meas, objects, plot: bool = True):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)
    F = array([[1.0, 0.0, 1.0, 0.0], 
               [0.0, 1.0, 0.0, 1.0], 
               [0.0, 0.0, 1.0, 0.0], 
               [0.0, 0.0, 0.0, 1.0]])
    H = array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
    Q = 0.06*eye(4)
    R = 0.03*eye(2)
    birth_belief = phd_BirthModels(10, 10)

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
            phd.prune(array([0.1]), array([3]), 20)
            pos_phd.append(phd.extract(0.1))
            #print(phd.extract())
            #print('--------------')

        pos_phd_all[m-1] = pos_phd
    

    if plot:
        K = np.arange(len(meas))
        # x-y-Raum
        #------------------------------------
        for m in range(1, len(pos_phd_all)+1):
            ax = plt.subplot(num, 1, m)
            for i in K:
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')
                #Objekte
                for j in range(len(objects[i])):
                    plt.plot(objects[i][j][0],objects[i][j][1],'ro',color='green')

                for l in range(len(pos_phd_all[m-1][i])):  
                    plt.plot(pos_phd_all[m-1][i][l][0],pos_phd_all[m-1][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('Clutter intensity is: '+str(m*ini))
        plt.suptitle('x-y-Raum für Variation der Clutter intensity')
        plt.axis([0,50,0,50])
        plt.show()

    return pos_phd_all



def varBirthNum_phd(ini: int, num: int, meas, objects, plot: bool = True):
    pos_phd_all: ndarray[List[ndarray]] = [None] * (num)

    F = array([[1.0, 0.0, 1.0, 0.0], 
               [0.0, 1.0, 0.0, 1.0], 
               [0.0, 0.0, 1.0, 0.0], 
               [0.0, 0.0, 0.0, 1.0]])
    H = array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
    Q = 0.1*eye(4)
    R = 0.05*eye(2)
    for m in range (1,num+1):   

        birth_belief = phd_BirthModels(ini*m, ini*m)

        survival_rate = 0.99
        detection_rate = 0.9
        intensity = 0.05

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
            phd.prune(array([0.1]), array([3]), 20)
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
            for i in K:
            #Schätzungen
  
                #Messungen
                for j in range(len(meas[i])):
                    plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

                #Objekte
                for j in range(len(objects[i])):
                    plt.plot(objects[i][j][0],objects[i][j][1],'ro',color='green')

                for l in range(len(pos_phd_all[m-1][i])):
                    #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')   
                    plt.plot(pos_phd_all[m-1][i][l][0],pos_phd_all[m-1][i][l][1],'ro',color= 'red', ms= 3)
                    ax.title.set_text('BirthmodelNumber is: '+str(m*ini))

        plt.suptitle('x-y-Raum for Variable Birthmodel Number')
        plt.axis([0,50,0,50])
        plt.show()

    return pos_phd_all

######################################################
phd_QRvar = varQR_phd(0.01, 5, meas, objects, True)
######################################################

'PRuning!!! wenn auf 50 dann schätzung wesentlich schlechter!!!'


######################################################
varIntens = varIntensity_phd(0.01, 5, meas, objects)
######################################################

######################################################
varBirth = varBirthNum_phd(2, 5, meas, objects)
######################################################
