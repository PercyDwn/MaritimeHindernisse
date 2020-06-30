import  matplotlib.pyplot as plt
import numpy as np
import math
from KalmanFilter import KalmanFilter

#Initialisierung
p_d = 0.99 #Detektionsrate
lambda_c =0.3 #Clutter Intensität
V= 5 #Cluttering Volume
T= 0.001 #Abtastzeit
F = [[1,T],
     [0,1]] #Systemmatrix
H = [1,0] #Ausgangsmatrix
n = 2 #Anzahl Objekte 
variance_motion = 0.25
variance_measurement = 0.2
variance_prior = 0.36
init_x1 = 0
init_x2 = 10
x = np.arange(-20,20,0.1)
init_prior_x1 = gaussian(x,init_x1,variance_prior)
init_prior_x2 = gaussian(x,init_x2,variance_prior)

k=1 #number of measurments 1-7
Pini = np.diag([.01, .009])
Q = [[.1,0],
     [0,.1]]
R = 10
#Kalmanfilter initialisieren
KF = KalmanFilter(dim_x=2, dim_z=1)
KF.x = np.array[0,1] # pos, velo init
KF.F = F
KF.H = H
KF.P = Pini
KF.Q = Q
KF.R = R

