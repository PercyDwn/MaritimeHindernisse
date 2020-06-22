import  matplotlib.pyplot as plt
import numpy as np
import math
def gnn(measurements,K,p_d,lambda_c,F,H,n):
    for k in K:
        m= len(measurements[k]) #Anzahl Messungen pro Zeitschritt k
        measurement_k = measurements[k] #Messung pro Zeitschritt k
        L_detection = np.zeros((n,m)) #Kostfunktion detektiert
        L_missdetection = np.zeros((n,n)) #Kostfunktion nicht-detektiert


        for i in range(n):
        #Prediktion
            #my,R,P=KalmanFilter.predict()
            R=1
            my =[[0.5],[0.5]]
            P = [[1,0],[0,1]]
        #Erzeugen Kostmatrix
            S = R+ np.matmul(H,np.matmul(P,np.transpose(H)))
            z_dach = np.matmul(H,my)
            L_missdetection[i][i] = np.log(1-p_d)
            for j in range(m):
                #if type(S) == 'int32':
                    L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(2*math.pi*S)-0.5*1/S*(measurement_k[i]-z_dach)*(z_dach-measurement_k[j]) # Achtung: gilt nur für eindimensionale S
                #else:
                #    L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(np.linalg.det(2*math.pi*S))-0.5*(np.matmul(np.transpose((measurement_k[i]-z_dach)),np.matmul(np.linalg.inv(S),(z_dach-measurement_k[j])))) # Achtung: gilt nur für eindimensionale S

            L = [L_detection,L_missdetection] #Gesamte Kostmatrix
         
            #Berechnen kombinatorische Matrix A mit Hungarian Algorithmus
     
        #Update
          #'KalmanFilter.update()'
      