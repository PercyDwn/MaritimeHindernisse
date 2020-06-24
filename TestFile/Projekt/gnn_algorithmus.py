import  matplotlib.pyplot as plt
import numpy as np
import math
from munkres import Munkres
def gnn(measurements,K,p_d,lambda_c,F,H,n):
    for k in K:
        hungarian = Munkres() # Objekt, welches den Hungarian Algorithmus darstellt
        m= len(measurements[k]) #Anzahl Messungen pro Zeitschritt k
        measurement_k = measurements[k] #Messung pro Zeitschritt k
        L_detection = np.zeros((n,m)) #Kostfunktion detektiert
        L_missdetection = np.zeros((n,n)) #Kostfunktion nicht-detektiert
        L_missdetection[np.arange(0,n,1)] = np.inf # Alle Einträge gleich unendlich setzen 
        L = np.zeros((n,m+n)) #Gesamte Kostenmatrix
        theta_k = np.zeros((1,n)) #Data Assossiation Vektor
        for i in range(n):
        #Prädiktion
            #my,R,P=KalmanFilter.predict()
            R=1
            my =[[0.5],[0.5]]
            P = [[1,0],[0,1]]
        #Kostenmatrix erzeugen 
            S = R+ np.matmul(H,np.matmul(P,np.transpose(H))) #Inovation Covariance
            z_hat = np.matmul(H,my) #Predicted detection
            L_missdetection[i][i] = np.log(1-p_d)
            for j in range(m):
                #if type(S) == 'int32':
                    L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(2*math.pi*S)-0.5*1/S*(measurement_k[j]-z_hat)*(measurement_k[j]-z_hat) # Achtung: gilt nur für eindimensionale S
                #else:
                #    L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(np.linalg.det(2*math.pi*S))-0.5*(np.matmul(np.transpose((measurement_k[i]-z_dach)),np.matmul(np.linalg.inv(S),(z_dach-measurement_k[j])))) # Achtung: gilt nur für eindimensionale S
        L = np.concatenate((L_detection,L_missdetection),axis=1) #Gesamte Kostenmatrix
        
        #Berechnen assignment Matrix A mit Hungarian Algorithmus
        indexes_opt = hungarian.compute(L) #Assignment matrix indexes
        # Berechnung von Data Assossiation theta_k
        for i in range(n):
            if indexes_opt[i][1]< m :
                theta_k[0][i] = indexes_opt[i][1] +1
            else :
                theta_k[i] =0
        
        
     
        #Update
          #'KalmanFilter.update()'
      