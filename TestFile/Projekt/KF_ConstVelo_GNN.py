import  matplotlib.pyplot as plt
import numpy as np
import math
import filterpy
#from KalmanFilter import KalmanFilter
from filterpy.kalman import KalmanFilter
from munkres import Munkres
from functions import gaussian 
from functions import createTestDataSet

#Initialisierung der DATEN
p_d = 0.99 #Detektionsrate
lambda_c =0.3 #Clutter Intensität
V= 5 #Cluttering Volume
T= 0.001 #Abtastzeit
F = np.array([[1,T],
     [0,1]]) #Systemmatrix
H = np.array([1,0]) #Ausgangsmatrix
n = 2 #Anzahl Objekte 
variance_motion = 0.25
variance_measurement = 0.2
variance_prior = 0.36
init_x1 = 0
init_x2 = 10
init_values =np.array([[init_x1,init_x2],[0, 0]])
x = np.arange(-20,20,0.1)
init_prior_x1 = gaussian(x,init_x1,variance_prior)
init_prior_x2 = gaussian(x,init_x2,variance_prior)

number_states = len(F) # Zuständezahl
estimate = np.zeros((number_states,n)) # Zustände geschätz 
estimate[0:number_states,0:n] = init_values #Anfangswerte hinzufügen
P = np.zeros((number_states,n*number_states)) 
'nullmatrix wirklich sinnvolll??? -> die Varianz des Schätzfehlers wird mit 0 also nicht vorhanden initialisiert großer wert wäre sinnvoller'


#Initialisierung KALMANFILTER
Pini = np.diag([.01, .009])
Q = np.array([[.1,0],
               [0,.1]])
R = np.array([[10]])

#KF = KalmanFilter(dim_x=2, dim_z=1)
#KF.x = np.array([0,1]) # pos, velo init
#KF.F = F    #Systemmatrix
#KF.H = H    #Ausgangsmatrix
#KF.P = Pini #Kovarianz des Schätzfehlers
#KF.Q = Q    #Kovarianz des Systemrauschens
#KF.R = R    #Kovarianz des Messrauschens
for i in range(n): 
                estimate_i = np.transpose(estimate[0:number_states,i] )   #Zustandände pro Objekt aus der gesamten estimates Matrix extraieren. Muss Transponiert werden, da Python mit stehenden Vektoren nicht umgehen kann
                P_i= P[0:number_states,i*number_states:number_states*(i+1)] #Kovarianz pro Objekt aus der gesamten P matrix extraieren 
            #Prädiktion mit Kalmanfilter
                estimate_i,P_i = filterpy.kalman.predict(estimate_i, P_i,F,Q)
                print('-----------------------------')
                print(estimate_i)
                print(P_i)
#x_k, P_k = filterpy.kalman.predict(init_values, Pini, F, Q)

#print(x_k)
#print(P_k)
#print('-----------------------------')
#print(estimate_i)
#print(P_i)