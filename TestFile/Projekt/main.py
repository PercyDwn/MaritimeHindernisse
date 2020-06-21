import  matplotlib.pyplot as plt
import numpy as np
import GNN 
from functions import gaussian

p_d = 0.99 #Detektionsrate
lambda_c =0.3 #Clutter Intensität
V= 5 #Cluttering Volume
T= 0.001 #Abtastzeit
A = [[1,T],
     [0,1]] #Systemmatrix
C = [1,0] #Ausgangsmatrix
n = 2 #Anzahl Objekte 
variance_motion = 0.25
variance_measurement = 0.2
variance_prior = 0.36
init_x1 = 0
init_x2 = 10
x = np.arange(-20,20,0.1)
init_prior_x1 = gaussian(x,init_x1,variance_prior)
init_prior_x2 = gaussian(x,init_x2,variance_prior)
def createTestDataSet():
        measurements = [[1,2.5,6,12],
             [3,3.5,4,13,13.5],
             [4.5,6,7, 14],
             [1,4,4.5,8,13,15],
             [4,4.5,5,6,16],
             [4,4.5,16,17,17.5,18],
             [1,2,3.5,3.6,15,15.5, 17,18]]
        objects = [[2.5,12],
             [3.5,13],
             [4.5, 14],
             [4.5 ,15],
             [4.5,16],
             [4,17],
             [3.5, 17]]
        K= np.arange(len(measurements))
        return measurements, objects,K

#Plots
measurements, objects,K = createTestDataSet()
for i in K:
    for j in range(len(measurements[i])):
        plt.plot(measurements[i][j],K[i],'ro',color='black')
    for j in range(len(objects[i])):
        plt.plot(objects[i][j],K[i],'ro',color='green')
plt.legend('Z_k','x_ist')     
plt.title('Messungen')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0,20,-1,len(K)+1])
plt.show()
