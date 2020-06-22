import  matplotlib.pyplot as plt
import numpy as np
import math
def gnn(measurements,K,p_d,lambda_c,F,H,n):
    for k in K:
        m= len(measurements[k]) #Anzahl Messungen pro Zeitschritt k
        L_detection = np.zeros(n,m) #Kostfunktion detektiert
        L_missdetection = np.zeros(n,n) #Kostfunktion nicht-detektiert


        for i in range(n):
        #Prediktion
            my,R,P=KalmanFilter.predict()
        #Erzeugen Kostmatrix
            S = R+ np.matmul(H,np.matmul(P,np.transpose(H)))
            z_dach = H*my
            L_missdetection[i][i] = np.log(1-Pd)
            for j in range(m):
                L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(np.linalg.det(2*math.pi))-0.5*(np.matmul(measurements[i]-z_dach,np.matmul(np.linalg.inv(S),(z_dach-measurements[j]))))
            L = np.concatenate(L_detection,L_missdetection) #Gesamte Kostmatrix
            #Berechnen kombinatorische Matrix A mit Hungarian Algorithmus

        #Update
          #'KalmanFilter.update()'
      