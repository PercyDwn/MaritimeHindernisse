import  matplotlib.pyplot as plt
import numpy as np
import pyrate_sense_filters_gmphd
import gnn_algorithmus
from functions import gaussian 
from functions import createTestDataSet
from functions import get_position
from pyrate_common_math_gaussian import Gaussian
from pyrate_sense_filters_gmphd import GaussianMixturePHD
from numpy import vstack
from numpy import array
from numpy import eye
import math
from MN import mnLogic

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
Q = 0.05*eye(2)
R = 0.05*eye(1)

#Def. Birth_belief
mean1 = vstack([1.0, 0.0])
covariance1 = array([[7.5, 0.0], [0.0, 1.0]])
mean2 = vstack([20.0, 1.0])
covariance2 = array([[7.5, 0.0], [0.0, 1.0]])
birth_belief = [Gaussian(mean1, covariance1), Gaussian(mean2, covariance2)]

survival_rate = 0.99
detection_rate = 0.99
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


phd.predict()
phd.correct([array([5.]), array([10.]), array([20.]) ])
position_phd = phd.extract()

print(position_phd)
print('--------------')

phd.predict()
phd.correct([array([6.]), array([21.]) ])
position_phd = phd.extract()

print(position_phd)
print('--------------')

phd.predict()
phd.correct([array([7.]), array([15.]), array([22.]) ])
position_phd = phd.extract()

print(position_phd)
print('--------------')

phd.predict()
phd.correct([array([8.]), array([23.]) ])
position_phd = phd.extract()

print(position_phd)
print('--------------')

phd.predict()
phd.correct([array([9.]), array([13.]), array([25.]) ])
phd.prune(array([0.05]), array([0.1]), 15)
position_phd = phd.extract()

print(position_phd)
print('--------------')

phd.predict()
phd.correct([array([10.]), array([20.]), array([26.]) ])
position_phd = phd.extract()

print(position_phd)
print('--------------')





#------------------------------------------------------------------------
#------------------------------------------------------------------------


#---------------------------------
#---------------------------------
#n = mnLogic(M,N,1,all_measurements) #Anzahl Objekte
#estimate_gnn = gnn_algorithmus.gnn(data,p_d,lambda_c,F,H,n,R,Q,init_values,P_i_init)
#position_gnn = get_position(estimate_gnn)
#print(position_gnn)

#print('............................................')
#print(real_objects)



#Plots


#for i in K:
#    for j in range(len(measurements[i])):
#        plt.plot(measurements[i][j],K[i]+1,'ro',color='black')
        
#    for j in range(n):
#        plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
#        plt.plot(position_gnn[i][j],K[i],'ro',color= 'orange')
#        plt.plot(position_gnn[-1][j],K[-1]+1,'ro',color= 'orange')
#plt.legend('Z_k','x_ist')     
#plt.title('Messungen')
#plt.xlabel('x')
#plt.ylabel('k')
#plt.axis([-15,30,-1,len(K)+1])
#plt.show()

