from ObjectHandler import *
import numpy as np 
from numpy.linalg import multi_dot
from ObjectHandler import *
import random 

#Externe Variablen
ObjectHandler = ObjectHandler()
N = 5
Q_horizon = [[0.1,0],
            [0,0.1]] ##Varianz des Modellrauschens Horizont
R_horizont = 1 
P_horizon = [[1,0],
            [0,1]]

H_horizon = [[1,0],
            [0,1]]#Ausgangsmatrix Horizont
F_horizon = [[1,0],
            [0,1]] #Systemmatrix Horizont



#k-schleife
def horizonState_gnn (ObjectHandler,N,R_horizon,P_horizon,H_horizon,F_horizon):
    pictures_availiable = True
    k = 0
    horizon_list = []
    while pictures_available == True:
        try:
             ObjectHandler.updateObjectStates()
             horizon_lines_k = ObjectHandler.getHorizon(k) #3 Horizont Kandidaten
        except InvalidTimeStepError as e:
               print(e.args[0])
               k = 0 
               pictures_availiable = False
               break
        if k<N:
            horizon_list.append(horizon_lines_k)
            #Datensatzladen
        if k==N:
            #MN Horizont aufrufen, um nur eine Linie von drei zu erhalten als Horizont-Kandidat
            est_hor_k = mn_horizon(horizon_list) #Estimate horizon am Zeitschritt k
        
        if k>=N:
            est_hor_k,P_horizon = kalman_filter_prediction(est_hor_k, P_horizon,F_horizon,Q_horizon) # Kalman Prädiktion
            H_heigth = [[1,0],
                        [0,0]] #Matrix extraiert nur die Höhe des Horizonts
            height_horizon = getHeight(horizon_lines_k)#Liste von Höhen aus Liste von Horizonten
            horizon_opt = min(abs(height_horizon-multi_dot([H_height,est_hor_k]))) #Horizon mit dem kleinsten Höheunterschied als optimal wählen
            if len(horizon_lines_k) == 0:# Analysieren ob Horizon detektiert wurde
                theta_k = 0
            else:
                theta_k =1
            est_hor_k,P_horizon = kalman_filter_update(est_hor_k,P_horizon,H_horizon,horizon_opt,theta_k,R_horizon,2)#Kalmann Update

            #Zustände auf gesamten Zustand zusammenführen
        k = k+1


def horizonState_phd (ObjectHandler,R_horizon,P_horizon,H_horizon,F_horizon):
    pictures_availiable = True
    k = 0
    horizon_list = []
    while pictures_availabel == True:
        try:
             ObjectHandler.updateObjectStates()
             horizon_lines_k = ObjectHandler.getHorizon(k) #3 Horizont Kandidaten
        except InvalidTimeStepError as e:
               print(e.args[0])
               k = 0 
               pictures_availiable = False
               break
        if k==0:
            random_index=random.randint(0, len(horizon_lines_k))
            est_hor_k = horizont_lines_k[random_index] #Bei der Initialisierung, einen zufälligen Horizontkandidaten von drei auswählen
        est_hor_k,P_horizon = kalman_filter_prediction(est_hor_k, P_horizon,F_horizon,Q_horizon) # Kalman Prädiktion
        H_heigth = [[1,0],
                    [0,0]] #Matrix extraiert nur die Höhe des Horizonts
        height_horizon = getHeight(horizon_lines_k)#Liste von Höhen aus Liste von Horizonten
        horizon_opt = min(abs(height_horizon-multi_dot([H_height,est_hor_k]))) #Horizon mit dem kleinsten Höheunterschied als optimal wählen
        if len(horizon_lines_k) == 0:# Analysieren ob Horizon detektiert wurde
           theta_k = 0
        else:
           theta_k =1
        est_hor_k,P_horizon = kalman_filter_update(est_hor_k,P_horizon,H_horizon,horizon_opt,theta_k,R_horizon,2)#Kalmann Update
        #Zustände auf gesamten Zustand zusammenführen

        k = k+1
 




  


def kalman_filter_prediction(estimates_i, P_i,F,Q):
   #Theorie Kalman Filter bei GNN: https://www.youtube.com/watch?v=MDMNsQJl6-Q&list=PLadnyz93xCLiCBQq1105j5Jeqi1Q6wjoJ&index=20 
    estimates_i = np.matmul(F,estimates_i) #Kalman Prädiktion estimates
    P_i =  multi_dot([F,P_i,np.transpose(F)]) +Q #Kalman Prädiktion 
    return estimates_i, P_i

def kalman_filter_update(estimate_i,P_i,H,z_opt_assossiation,theta_i,R,dimensions):
     help_K_1 =np.matmul(P_i,np.transpose(H)) #Hilfsvariable für die Berechnung von K. Muss Transponiert werden, da Python mit stehenden Vektoren nicht umgehen kann
     help_K_2 = lin.inv(np.matmul(H,help_K_1)+R) #Hilfsvariable für die Berechnung von K
     if theta_i != 0: #Wenn Objekt detektiert wurde => Kalmanprediktion durchführen
         
         help_estimate = z_opt_assossiation - np.matmul(H,estimate_i)


         if dimensions==1:
            K = help_K_1*help_K_2
            estimate_i = estimate_i +K*help_estimate #K Transponieren aufgrung Python und nicht der Theorie
            help_P_1 =K*np.matmul(H,P_i)
         else:
             K = np.matmul(help_K_1,help_K_2)
             estimate_i = estimate_i +np.matmul(K,help_estimate) #K Transponieren aufgrung Python und nicht der Theorie
             help_P_1 = multi_dot([K,H,P_i])
         P_i = P_i - help_P_1
         
     else:

         P_i = P_i
         estimate_i = estimate_i
     
     return estimate_i,P_i