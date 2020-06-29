import  matplotlib.pyplot as plt
import numpy as np
import math
from munkres import Munkres
def gnn(measurement_k,p_d,lambda_c,F,H,n,R,Q):
    #Algorithmus eignet sich nur für GNN mit einem linearen und gaußverteilten Modell 

        hungarian = Munkres() # Objekt, welches den Hungarian Algorithmus darstellt
        m= len(measurement_k) #Anzahl Messungen pro Zeitschritt k
        L_detection = np.zeros((n,m)) #Kostfunktion detektiert
        L_missdetection = np.zeros((n,n)) #Kostfunktion nicht-detektiert
        L_missdetection[np.arange(0,n,1)] = np.inf # Alle Einträge gleich unendlich setzen 
        L = np.zeros((n,m+n)) #Gesamte Kostenmatrix
        theta_k = np.zeros((1,n)) #Data Assossiation Vektor
        number_coordinates = int(len(F)/2) # Zahl Koordinaten: 1 wenn z= x, 2 wenn z=[x;y]
        expected_states = np.zeros((n,len(F))) # Zustände geschätz 
        total_cost = 0 #Kosten Data Assossiation 
        for i in range(n):
        #Prädiktion
            #my,P=KalmanFilter.predict()
            my = [[0.5],[0.5]]
            P = [[1,0],[0,1]]
        #Kostenmatrix erzeugen 
            S = R+ np.matmul(H,np.matmul(P,np.transpose(H))) #Inovation Kovarianz
            z_hat = np.matmul(H,my) #Predicted detection
            L_missdetection[i][i] = np.log(1-p_d)
            for j in range(m):
                if number_coordinates == 1:
                    L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(2*math.pi*S)-0.5*1/S*(measurement_k[j]-z_hat)#*(measurement_k[j]-z_hat) # Achtung: gilt nur für eindimensionale S
                else:
                    L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(np.linalg.det(2*math.pi*S))-0.5*(np.matmul(np.transpose((measurement_k[j]-z_dach)),np.matmul(np.linalg.inv(S),(z_dach-measurement_k[j])))) 
        
        L= np.concatenate((L_detection,L_missdetection),axis=1)
        print(L)
        #Berechnen assignment Matrix A mit Hungarian Algorithmus
        indexes_opt = hungarian.compute(L) #Assignment matrix indexes
        print(L)
        # Berechnung von Data Assossiation theta_k
        for i in range(n):
            #print(L)
            total_cost += L[i][indexes_opt[i][1]]
            weight_opt = np.exp(-total_cost)
            if indexes_opt[i][1]< m :
                theta_k[0][i] = indexes_opt[i][1] +1
                if number_coordinates ==1 :
                   z_opt_assossiation = measurement_k[indexes_opt[i][1]] # Messung der wahrscheinlichsten Hypothese
                else:
                    z_opt_assossiation = measurement_k[np.arange(0,number_coordinates,1)][indexes_opt[i][1]]

                
            else :
                theta_k[0][i] = 0
        a=1
            #expected_states[i][np.arange(0,number_coordinates,1)] = KalmanFilter.update()
            
     
        #Update
          #'KalmanFilter.update()'
      