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

#Test Datensatz
warmup_data, measurements, real_objects,K = createTestDataSet()


#GM_PHD filter erzeugen
#------------------------------------------------------------------------

#Our belief of how targets are generetaded is for them to start with a position and velocity of 0.
mean = vstack([1.0, 0.0])
covariance = array([[1.0, 0.0], [0.0, 1.0]])
birth_belief = [Gaussian(mean, covariance)]

#We need to tell the filter how certain we are to detect targets and whether they survive. Also, the amount of clutter in the observed environment is quantized.
survival_rate = 0.99
detection_rate = 0.99
intensity = 0.01

#GM-PHD Aufruf
#------------------------------------------------------------------------

data = measurements[:]
all_measurements = np.concatenate((warmup_data,measurements),axis=0).tolist()
estimate_all =[]
estimate_all.append(init_values.tolist()) #Liste mit  Erwartungswerten von allen Zuständen aller Objekten über alle Zeitschritten
k = 1   #Zeitschritt


#while len(data)>0: #While: data nicht leer
#    measurement_k = data.pop(0)  #Erste Messung aus Datensatz (wird danach aus Datenliste entfernt)
#    phd.predict()
#    phd.correct(np.array(measurement_k))
#    position_phd = phd.extract()
#    estimate_all.append(position_phd.tolist())
#    k = k+1

#------------------------------------------------------------------------
#------------------------------------------------------------------------

F = array([[1.0, 1.0], 
           [0.0, 1.0]])
H = array([[1.0, 0.0]])
Q = 0.1*eye(2)
R = 1.*eye(1)

#Def. Birth_belief
mean1 = vstack([1.0, 0.0])
covariance1 = array([[7.5, 0.0], [0.0, 1.0]])
mean2 = vstack([30.0, 1.0])
covariance2 = array([[5., 0.0], [0.0, 1.0]])
birth_belief = [Gaussian(mean1, covariance1), Gaussian(mean2, covariance2)]

survival_rate = 0.999
detection_rate = 0.8
intensity = 0.04

phd = GaussianMixturePHD(
                birth_belief,
                survival_rate,
                p_d,
                intensity,
                F,
                H,
                Q,
                R)


meas: List[ndarray] = []
pos_phd: List[ndarray] = []
meas.append([array([5.]), array([30.]), array([10.]), array([30.]), array([-10.]), array([50*random.random()]), array([50*random.random()]), array([-15*random.random()])])
meas.append([array([6.]), array([31.]), array([9.]), array([15.]), array([30.]), array([50*random.random()]), array([50*random.random()]), array([-15*random.random()])])
meas.append([array([7.]), array([32.]), array([15.]), array([25.]), array([50*random.random()]), array([50*random.random()]), array([-15*random.random()])])
meas.append([array([8.]), array([33.]), array([40.]), array([45.]), array([-9.]), array([50*random.random()]), array([50*random.random()]), array([-15*random.random()])])
meas.append([array([9.]), array([20.]), array([35.]), array([19.]), array([-7.]), array([50*random.random()]), array([50*random.random()]), array([-15*random.random()])])
meas.append([array([10.]), array([21.]), array([36.]), array([-14.]), array([50*random.random()]), array([50*random.random()]), array([-15*random.random()])])
meas.append([array([10.]), array([22.]), array([37.]), array([-6.]), array([50*random.random()]), array([50*random.random()]), array([-15*random.random()])])
meas.append([array([12.]), array([20.]), array([38.]), array([26.]), array([-8.]), array([-9.]), array([50*random.random()]), array([50*random.random()])])


for z in meas:
    phd.predict()
    phd.correct(z)
    pos_phd.append(phd.extract())
    #print(phd.extract())
    #print('--------------')
    #pruning
    phd.prune(array([0.4]), array([3]), 10)

#Plots
# ------------------------
K = np.arange(len(meas))
real1 = np.arange(5., 13., 1.)
real2 = np.arange(30., 38., 1.)
plt.plot(real1, K, color= 'orange')
plt.plot(real2, K, color= 'orange')

for i in K:
    #Messungen
    for j in range(len(meas[i])):
        plt.plot(meas[i][j],K[i],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        #plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
        plt.plot(pos_phd[i][l][0],K[i],'ro',color= 'red', ms= 3)
        
plt.legend(['Z_k', 'phd_k'])     
plt.title('Messungen')
plt.xlabel('x')
plt.ylabel('k')
plt.axis([-15,50,-1,len(K)+1])
plt.show()


#phd.predict()
#phd.correct([array([5.]), array([10.]), array([20.])])
#position_phd = phd.extract()

##print(position_phd)
#print('--------------')

#phd.predict()
#phd.correct([array([6.]), array([21.])])
#position_phd = phd.extract()

##print(position_phd)
#print('--------------')

#phd.predict()
#phd.correct([array([7.]), array([15.]), array([22.])])
#position_phd = phd.extract()

##print(position_phd)
#print('--------------')

#phd.predict()
#phd.correct([array([8.]), array([23.]) ])
#position_phd = phd.extract()

##print(position_phd)
#print('--------------')

#phd.predict()
#phd.correct([array([9.]), array([20.]), array([25.]) ])
#phd.prune(array([0.05]), array([0.1]), 15)
#position_phd = phd.extract()

##print(position_phd)
#print('--------------')

#phd.predict()
#phd.correct([array([10.]), array([21.]), array([26.]) ])
#position_phd = phd.extract()

##print(position_phd)
#print('--------------')

#phd.predict()
#phd.correct([array([10.]), array([22.]), array([27.]) ])
#position_phd = phd.extract()
##print(position_phd)
#print('--------------')

#phd.predict()
#phd.correct([array([11.]), array([20.]), array([28.]) ])
#position_phd = phd.extract()
#phd.prune(array([0.05]), array([0.1]), 15)

##print(position_phd)
#print('--------------')

