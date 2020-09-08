import  matplotlib.pyplot as plt
import numpy as np
import pyrate_sense_filters_gmphd
import gnn_algorithmus
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
from MN import mnLogic
# Typing
from typing import List

#Initialisierung
#------------------------------------------------------------------------

p_d = 0.95 #Detektionsrate
lambda_c =0.3 #Clutter Intensität
V= 5 #Cluttering Volume
T= 0.1 #Abtastzeit
F = array([[1,T],
     [0,1]]) #Systemmatrix
H = np.array([1,0]) #Ausgangsmatrix 
Q = array([[100 ,0],[0 ,100]]) #Varianz des Modellrauschens
R = array([[10]]) #Varianz des Messrauschens
init_x1 = -1
init_x2 = 15 
init_values =np.array([[init_x1,init_x2],[0, 0]])
P_i_init = [[10,0],[0,10]]
M = 4 #Anzahl der benoetigten Detektionen
N= 5 #Anzahl der Scans



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
Q = 0.1*eye(4)
R = 0.01*eye(2)

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

survival_rate = 0.999
detection_rate = 0.9
intensity = 0.01

phd = GaussianMixturePHD(
                birth_belief,
                survival_rate,
                p_d,
                intensity,
                F,
                H,
                Q,
                R)

objects: List[ndarray] = []
meas: List[ndarray] = []
pos_phd: List[ndarray] = []

#Objekte erstellen
#-------------------------------
for i in range(10):
    objects.insert(i, [array([[0.+i], [10.+i]]), array([[50.-2*i], [15.]]), array([[50.-i], [15.+i]])] )
   

for i in range(20):
    objects.insert(10+i, [array([[10.+1.5*i], [20.+i]]), array([[30.-2*i], [15.]]), array([[40.-i], [26.+i]])] )
    

#Messdaten erstellen
#-------------------------------
for i in range(10):
    meas.insert(i, [array([[0.+i], [10.+i]]), array([[50.-2*i], [15.]]), array([[50.-i], [15.+i]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]) ] )
  

for i in range(20):
    meas.insert(10+i, [ array([[10.+1.5*i], [20.+i]]), array([[30.-2*i], [15.]]), array([[40.-i], [26.+i]]), array([[50*random.random()], [50*random.random()]]),  array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]) ] )
    



#PHD-Filter anwenden
#-----------------------------
for z in meas:
    phd.predict()
    phd.correct(z)
    pos_phd.append(phd.extract())
    print(phd.extract())
    print('--------------')
    #pruning
    phd.prune(array([0.3]), array([3]), 10)

#Plots
# ------------------------
# ------------------------

K = np.arange(len(meas))
#real1 = np.arange(5., 13., 1.)
#real2 = np.arange(30., 38., 1.)
#plt.plot(real1, K, color= 'orange')
#plt.plot(real2, K, color= 'orange')

## x-Koordinate
##------------------------------------
for i in K:
    #Messungen
    for j in range(len(meas[i])):
        plt.plot(meas[i][j][0],K[i],'ro',color='black')
    
    #Objekte
    for j in range(len(objects[i])):
        plt.plot(objects[i][j][0],K[i],'ro',color='green')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
        plt.plot(pos_phd[i][l][0],K[i],'ro',color= 'red', ms= 3)
        
plt.legend('Zk', 'phd')     
plt.title('x-Koordinate')
plt.xlabel('x')
plt.ylabel('k')
plt.axis([-5,55,-1,len(K)+1])
plt.show()


## y-Koordinate
##------------------------------------
for i in K:
    #Messungen
    for j in range(len(meas[i])):
        plt.plot(meas[i][j][1],K[i],'ro',color='black')

    
    #Objekte
    for j in range(len(objects[i])):
        plt.plot(objects[i][j][1],K[i],'ro',color='green')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
        plt.plot(pos_phd[i][l][1],K[i],'ro',color= 'red', ms= 3)
        
plt.legend(['Zk', 'phd'])     
plt.title('y-Koordinate')
plt.xlabel('y')
plt.ylabel('k')
plt.axis([-5,55,-1,len(K)+1])
plt.show()



# x-y-Raum
#------------------------------------

for i in K:
    #Messungen
    for j in range(len(meas[i])):
        plt.plot(meas[i][j][0],meas[i][j][1],'ro',color='black')

    #Objekte
    for j in range(len(objects[i])):
        plt.plot(objects[i][j][0],objects[i][j][1],'ro',color='green')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
        plt.plot(pos_phd[i][l][0],pos_phd[i][l][1],'ro',color= 'red', ms= 3)
        
plt.legend(['Zk', 'phd'])     
plt.title('x-y-Raum')
plt.xlabel('x-Koord.')
plt.ylabel('y-Koord.')
plt.axis([-5,55,-5,55])
plt.show()

