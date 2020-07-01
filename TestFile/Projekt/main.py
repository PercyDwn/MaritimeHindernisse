import  matplotlib.pyplot as plt
import numpy as np
import gnn_algorithmus
from functions import gaussian 
from functions import createTestDataSet
import math

#Initialisierung
p_d = 0.99 #Detektionsrate
lambda_c =0.3 #Clutter Intensität
V= 5 #Cluttering Volume
T= 0.001 #Abtastzeit
F = [[1,T],
     [0,1]] #Systemmatrix
H = [1,0] #Ausgangsmatrix
n = 2 #Anzahl Objekte 
Q = [[0.25 ,0],[0 ,0.25]] #Varianz des Modellrauschens
R = 0.2 #Varianz des Messrauschens
variance_prior = 0.36
init_x1 = 0
init_x2 = 10
x = np.arange(-20,20,0.1)
init_prior_x1 = gaussian(x,init_x1,variance_prior)
init_prior_x2 = gaussian(x,init_x2,variance_prior)

#Test Datensatz
measurements, real_objects,K = createTestDataSet()
#GNN Aufruf
estimate_gnn = gnn_algorithmus.gnn(measurements,p_d,lambda_c,F,H,n,R,Q)

     

#Plots

for i in K:
    for j in range(len(measurements[i])):
        plt.plot(measurements[i][j],K[i],'ro',color='black')
    for j in range(len(real_objects[i])):
        plt.plot(real_objects[i][j],K[i],'ro',color='green')
plt.legend('Z_k','x_ist')     
plt.title('Messungen')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0,20,-1,len(K)+1])
plt.show()
