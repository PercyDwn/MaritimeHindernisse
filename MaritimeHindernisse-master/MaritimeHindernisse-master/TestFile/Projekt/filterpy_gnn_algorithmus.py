import numpy as np
from numpy import linalg as lin
import math
import filterpy
from munkres import Munkres



# Theorie GNN : https://www.youtube.com/watch?v=MDMNsQJl6-Q&list=PLadnyz93xCLiCBQq1105j5Jeqi1Q6wjoJ&index=21&t=0s
def gnn(data,p_d,lambda_c,F,H,n,R,Q,init_values):
    #data Messung aller Zeitschritten
    #p_d =detection rate
    #lambda_c Clutter Intensität
    #F, H  System bzw Ausgangsmatrux
    #n Anzahl Objekten
    #R,Q Kovarianz Matrizen
    #Algorithmus eignet sich nur für GNN mit einem linearen und gaußverteilten Modell 
         #Initialisierung
        hungarian = Munkres() # Objekt, welches den Hungarian Algorithmus darstellt
        number_states = len(F) # Zuständezahl
        theta_k = np.zeros((1,n)) #Data Assossiation Vektor
        number_coordinates = int(number_states/2) # Zahl Koordinaten: 1 wenn z= x, 2 wenn z=[x;y]
        estimate = np.zeros((number_states,n)) # Zustände geschätz 
        estimate[0:number_states,0:n] = init_values #Anfangswerte hinzufügen
        P = np.zeros((number_states,n*number_states)) #Kovarianzmatrix des Schätzfehlers
        #total_cost = 0 #Kosten Data Assossiation 
        estimate_all =[]
        estimate_all.append(init_values.tolist()) #Liste mit  Erwartungswerten von allen Zuständen aller Objekten über alle Zeitschritten
        z_opt_assossiation= 0 # Messung der wahrscheinlichsten Hypothese
        
            
        ## Berechnung mit Messdaten/Testdaten
        while len(data)>0: #While: data nicht leer
            measurement_k = data.pop(0)  #Erste Messung aus Datensatz (wird danach aus Datenliste entfernt)
            m= len(measurement_k) #Anzahl Messungen pro Zeitschritt k
            L_detection = np.zeros((n,m)) #Kostfunktion detektiert
            L_missdetection = np.zeros((n,n)) #Kostfunktion nicht-detektiert
            L_missdetection[:,:] = np.inf # Alle Einträge gleich unendlich setzen 
            L = np.zeros((n,m+n)) #Gesamte Kostenmatrix
            
            for i in range(n):
                estimate_i = np.transpose(estimate[0:number_states,i] )   #Zustandände pro Objekt aus der gesamten estimates Matrix extraieren. Muss Transponiert werden, da Python mit stehenden Vektoren nicht umgehen kann
                P_i= P[0:number_states,i*number_states:number_states*(i+1)] #Kovarianz pro Objekt aus der gesamten P matrix extraieren 
            #Prädiktion mit Kalmanfilter
                estimate_i,P_i = kalman_filter_prediction(estimate_i, P_i,F,Q) 
         
            #Kostenmatrix erzeugen 
                S = R+ np.matmul(H,np.matmul(P_i,np.transpose(H))) #Inovation Kovarianz
                z_hat = np.matmul(H,estimate_i) #Predicted detection
                L_missdetection[i][i] = np.log(1-p_d)
                for j in range(m):
                    if number_coordinates == 1:
                        
                        L_detection[i][j] =  np.log(p_d/lambda_c) - 0.5*np.log(2*math.pi*abs(S))-0.5*1/S*(measurement_k[j]-z_hat)*(measurement_k[j]-z_hat)
                        
                         
                    else:
                        L_detection[i][j] = np.log(p_d/lambda_c) - 0.5*np.log(np.linalg.det(2*math.pi*S))-0.5*(np.matmul(np.transpose((measurement_k[j]-z_dach)),np.matmul(np.linalg.inv(S),(z_dach-measurement_k[j])))) 
            
                 
            L= np.concatenate((L_detection,L_missdetection),axis=1) #L_detection und L_missdetection zusammensetzen
            
            #Berechnen assignment Matrix A mit Hungarian Algorithmus
            indexes_opt = hungarian.compute(L) #Assignment matrix indexes
           
            # Berechnung von Data Assossiation theta_k
            for i in range(n):
            
                #total_cost += L[i][indexes_opt[i][1]]
                #weight_opt = np.exp(-total_cost)
                #Fallunterscheidung: theta = index von Messung wenn die Detektion einem Objekt entspicht, theta = 0 wenn die Detektion einem Clutter entspricht
                if indexes_opt[i][1]< m :
                    theta_k[0][i] = indexes_opt[i][1] +1
                    if number_coordinates ==1 :
                       z_opt_assossiation = measurement_k[indexes_opt[i][1]] # Messung der wahrscheinlichsten Hypothese
                    else:
                       z_opt_assossiation = measurement_k[np.arange(0,number_coordinates,1)][indexes_opt[i][1]]
                
                else:
                    theta_k[0][i] = 0
            
                

                estimate_i,P_i = kalman_filter_update(estimate_i,P_i,H,z_opt_assossiation,theta_k[0][i],R,number_coordinates) #Update P und estimate_i mit Kalman-Korrekturschritt
                P[0:number_states,i*number_states:number_states*(i+1)] = P_i #P_i in die gesamte P Matrix wieder einfügen
                estimate[0:number_states,i] = estimate_i #estimates_i in die gesamte estimates Matrix wieder einfügen
            
            estimate_all.append(estimate.tolist())
        
        
        return estimate_all    
            
     
        
def kalman_filter_prediction(estimates_i, P_i,F,Q):
    estimates_i, P_i = filterpy.kalman.predict(estimates_i, P_i,F,Q) #Kalman Prädiktion estimates und P
    return estimates_i, P_i

def kalman_filter_update(estimate_i,P_i,H,z_opt_assossiation,theta_i,R,number_coordinates):
    if theta_i != 0: #Wenn Objekt detektiert wurde => Kalmanprediktion durchführen
        estimate_i, P_i = filterpy.kalman.update(estimate_i, P_i,z_opt_assossiation,R, H) #Kalman Korrektur für estimates und P
    else: #Objekt nicht erkannt => Werte durchreichen
        P_i = P_i
        estimate_i = estimate_i
    return estimate_i,P_i