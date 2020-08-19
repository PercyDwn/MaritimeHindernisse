import numpy as np
import  matplotlib.pyplot as plt
from numpy import linalg as lin
import math
from munkres import Munkres
from numpy.linalg import multi_dot #Matrix Mult. mit mehreren Matrizen
from functions import initialize_values
from MN import *
from ObjectHandler import *
# Theorie GNN : https://www.youtube.com/watch?v=MDMNsQJl6-Q&list=PLadnyz93xCLiCBQq1105j5Jeqi1Q6wjoJ&index=21&t=0s
def gnn(p_d,M,N,dimensions,T,ObjectHandler,Q,R,P_i_init):
    #data Messung aller Zeitschritten
    #p_d =detection rate
    #lambda_c Clutter Intensität
    #F, H  System bzw Ausgangsmatrux
    #n Anzahl Objekten
    #R,Q,P_i_init Kovarianz Matrizen
    #Algorithmus eignet sich nur für GNN mit einem linearen und gaußverteilten Modell 
        F = [[1,T,0,0],
             [0,1,0,0],
             [0,0,1,T],
              [0,0,0,1]] #Systemmatrix 
        H =[[1,0,0,0],
            [0,0,1,0]]#Ausgangsmatrix
        warmup_data = []
        ObjectHandler.updateObjectStates()
        # Bereitet die Daten für den Aufruf vom M/N Algorithmus, indem die ersten N Zeitschritten als Warmlaufdaten genutz werden
        for k in range(N):   
            
            warmup_data.append(ObjectHandler.getObjectStates(k)) #Daten der Detektion über alle Zeitschritten
           
                 
         #Initialisierung
        mn_data = warmup_data[:] #Daten für M/N Algorithmus
        n = 2
        #n = mnLogic(M,N,1,mn_data) #Anzahl Objekte
        _,_,_,_, init_values,_ = initialize_values(dimensions,T,n,data[0]) #Initialisierung aller Anfangswerten 

        hungarian = Munkres() # Objekt, welches den Hungarian Algorithmus darstellt
        number_states = len(F) # Zuständezahl
        theta_k = np.zeros((1,n)) #Data Assossiation Vektor
        estimate = np.zeros((number_states,n)) # Zustände geschätz 
        estimate[0:number_states,0:n] = init_values #Anfangswerte hinzufügen
        P = np.zeros((number_states,n*number_states))
        for i in range(n):
            P[0:number_states,i*number_states:number_states*(i+1)] = P_i_init #Kovarianzmatrix des Schätzfehlers
          
        
        estimate_all =[]
        estimate_all.append(init_values.tolist()) #Liste mit  Erwartungswerten von allen Zuständen aller Objekten über alle Zeitschritten
        k = 0   #Zeitschritt
        fig = plt.figure()

        
        
            
        ## Berechnung auf Messdaten
        while ObjectHandler.getObjectStates(k+N) !=[]: #While: data nicht leer
            
            measurement_k = ObjectHandler.getObjectStates(k+N) #Daten der Detektion eines Zeitschrittes  
            m= len(measurement_k) #Anzahl Messungen pro Zeitschritt k
            lambda_c = 0.001 + 1-n/m  #Clutter Intensität
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
            # Theorie zur Kostenmatrix :https://www.youtube.com/watch?v=uwBVssiFpOg&list=PLadnyz93xCLiCBQq1105j5Jeqi1Q6wjoJ&index=17&t=0s
                S = R+ multi_dot([H,P_i,np.transpose(H)]) #Inovation Kovarianz
                S_skaliert = (2*math.pi*S).tolist() # Mit 2*pi skaliertes S
                S = S.tolist() #Detreminate und Inverse von List sklara möglich (array skalar unmöglich)
                z_hat = np.matmul(H,estimate_i) #Predicted detection
                L_missdetection[i][i] = -np.log(1-p_d)
                
                for j in range(m):
                    help_L_0 = []
                    help_L_0.append(np.array(measurement_k[j])-z_hat) #Hilfsvariable für die Berechnung von L (muss eine Liste StopAsyncIteration)
                    help_L_1 = -0.5*(multi_dot([(help_L_0),np.linalg.inv(S),np.transpose(help_L_0)])) 
                    help_L_2 = - 0.5*np.log(np.linalg.det(S))
                    help_L_3  = np.log(p_d/lambda_c)
                    L_detection[i][j] = -(help_L_3 + help_L_2+ help_L_1)
                estimate[0:number_states,i] = estimate_i #estimates_i in die gesamte estimates Matrix wieder einfügen   
                P[0:number_states,i*number_states:number_states*(i+1)] = P_i #P_i in die gesamte P Matrix wieder einfügen
            L= np.concatenate((L_detection,L_missdetection),axis=1) #L_detection und L_missdetection zusammensetzen
            
            #Berechnen assignment Matrix A mit Hungarian Algorithmus
 
            indexes_opt = hungarian.compute(np.concatenate((L_detection,L_missdetection),axis=1)) #Assignment matrix indexes

           
            # Berechnung von Data Assossiation theta_k
            for i in range(n):
            

                

                #Fallunterscheidung: theta = index von Messung wenn die Detektion einem Objekt entspicht, theta = 0 wenn die Detektion einem Clutter entspricht
                estimate_i = np.transpose(estimate[0:number_states,i] )   #Zustandände pro Objekt aus der gesamten estimates Matrix extraieren. Muss Transponiert werden, da Python mit stehenden Vektoren nicht 
                P_i= P[0:number_states,i*number_states:number_states*(i+1)] #Kovarianz pro Objekt aus der gesamten P matrix extraieren 
                index_opt = indexes_opt[i][1]
                z_opt_assossiation = 0 #Initialisierung von : Messung der wahrscheinlichsten Hypothese
                if index_opt< m :
                    theta_k[0][i] = index_opt +1

                    if dimensions ==1 :
                       z_opt_assossiation = measurement_k[index_opt] # Messung der wahrscheinlichsten Hypothese
                    else:
                       z_opt_assossiation = measurement_k[index_opt]
                    
                
                else :
                    theta_k[0][i] = 0
                    
                

                estimate_i,P_i = kalman_filter_update(estimate_i,P_i,H,z_opt_assossiation,theta_k[0][i],R,dimensions) #Update P und estimate_i mit Kalman-Korrekturschritt
                position_i = np.matmul(H,estimate_i) #Position eines Objekts aus den Zuständen 
                P[0:number_states,i*number_states:number_states*(i+1)] = P_i #P_i in die gesamte P Matrix wieder einfügen
                estimate[0:number_states,i] = estimate_i #estimates_i in die gesamte estimates Matrix wieder einfügen
                print(position_i)
                

                #Echtzeit Plots 
                plt.plot(position_i[0],position_i[1],"x",color= 'orange')
                   
                plt.title('Geschätze Position')
                plt.xlabel('x_koordinate')
                plt.ylabel('y_kordinate')
                
                
            estimate_all.append(estimate.tolist())
            k = k+1
            mn_data.pop(0) #Löschen ältestes Element
            mn_data.append(measurement_k) #Aktuelle Messung einfügen
            n = 2
            #n = mnLogic(M,N,1,mn_data) #Anzahl Objekte
        plt.grid()
        plt.show()   
        return estimate_all ,n   
            
     
#!nicht einheitlich mit estimateS_i! -> Fehler ??
        
def kalman_filter_prediction(estimates_i, P_i,F,Q):
   #Theorie Kalman Filter bei GNN: https://www.youtube.com/watch?v=MDMNsQJl6-Q&list=PLadnyz93xCLiCBQq1105j5Jeqi1Q6wjoJ&index=20 
    estimates_i = np.matmul(F,estimates_i) #Kalman Prädiktion estimates
    P_i =multi_dot([F,P_i,np.transpose(F)]) +Q #Kalman Prädiktion 
    
    
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



#Berechnung auf Testdaten
def gnn_testdaten(p_d,M,N,dimensions,T):
    #data Messung aller Zeitschritten
    #p_d =detection rate
    #lambda_c Clutter Intensität
    #F, H  System bzw Ausgangsmatrux
    #n Anzahl Objekten
    #R,Q,P_i_init Kovarianz Matrizen
    #Algorithmus eignet sich nur für GNN mit einem linearen und gaußverteilten Modell 
         #Initialisierung
        warmup_data,data,real_object,K = createTestDataSet(dimensions)
        mn_data = warmup_data[:] #Daten für M/N Algorithmus
        n = 2
        #n = mnLogic(M,N,1,mn_data) #Anzahl Objekte
        F,H,Q,R, init_values,P_i_init = initialize_values(dimensions,T,n,data[0]) #Initialisierung aller Anfangswerten 

        hungarian = Munkres() # Objekt, welches den Hungarian Algorithmus darstellt
        number_states = len(F) # Zuständezahl
        theta_k = np.zeros((1,n)) #Data Assossiation Vektor
        estimate = np.zeros((number_states,n)) # Zustände geschätz 
        estimate[0:number_states,0:n] = init_values #Anfangswerte hinzufügen
        P = np.zeros((number_states,n*number_states))
        for i in range(n):
            P[0:number_states,i*number_states:number_states*(i+1)] = P_i_init #Kovarianzmatrix des Schätzfehlers
          
        
        estimate_all =[]
        estimate_all.append(init_values.tolist()) #Liste mit  Erwartungswerten von allen Zuständen aller Objekten über alle Zeitschritten
        k = 1   #Zeitschritt
       

        
        
            
        ## Berechnung mit Messdaten/Testdaten
        while len(data)>0: #While: data nicht leer
            
            measurement_k = data.pop(0)  #Erste Messung aus Datensatz (wird danach aus Datenliste entfernt)
            total_cost = 0 #Kosten Data Assossiation 
            m= len(measurement_k) #Anzahl Messungen pro Zeitschritt k
            lambda_c = 0.001 + 1-n/m  #Clutter Intensität
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
            # Theorie zur Kostenmatrix :https://www.youtube.com/watch?v=uwBVssiFpOg&list=PLadnyz93xCLiCBQq1105j5Jeqi1Q6wjoJ&index=17&t=0s
                S = R+ multi_dot([H,P_i,np.transpose(H)]) #Inovation Kovarianz
                S_skaliert = (2*math.pi*S).tolist() # Mit 2*pi skaliertes S
                S = S.tolist() #Detreminate und Inverse von List sklara möglich (array skalar unmöglich)
                z_hat = np.matmul(H,estimate_i) #Predicted detection
                L_missdetection[i][i] = -np.log(1-p_d)
                
                for j in range(m):
                    help_L_0 = []
                    help_L_0.append(np.array(measurement_k[j])-z_hat) #Hilfsvariable für die Berechnung von L (muss eine Liste StopAsyncIteration)
                    help_L_1 = -0.5*(multi_dot([(help_L_0),np.linalg.inv(S),np.transpose(help_L_0)])) 
                    help_L_2 = - 0.5*np.log(np.linalg.det(S))
                    help_L_3  = np.log(p_d/lambda_c)
                    L_detection[i][j] = -(help_L_3 + help_L_2+ help_L_1)
                estimate[0:number_states,i] = estimate_i #estimates_i in die gesamte estimates Matrix wieder einfügen   
                P[0:number_states,i*number_states:number_states*(i+1)] = P_i #P_i in die gesamte P Matrix wieder einfügen
            L= np.concatenate((L_detection,L_missdetection),axis=1) #L_detection und L_missdetection zusammensetzen
            
            #Berechnen assignment Matrix A mit Hungarian Algorithmus
 
            indexes_opt = hungarian.compute(np.concatenate((L_detection,L_missdetection),axis=1)) #Assignment matrix indexes

           
            # Berechnung von Data Assossiation theta_k
            for i in range(n):
            

                #total_cost += L_old[i][indexes_opt[i][1]]
                #weight_opt = np.exp(-total_cost)
                total_cost += L[i][indexes_opt[i][1]]

                #Fallunterscheidung: theta = index von Messung wenn die Detektion einem Objekt entspicht, theta = 0 wenn die Detektion einem Clutter entspricht
                estimate_i = np.transpose(estimate[0:number_states,i] )   #Zustandände pro Objekt aus der gesamten estimates Matrix extraieren. Muss Transponiert werden, da Python mit stehenden Vektoren nicht 
                P_i= P[0:number_states,i*number_states:number_states*(i+1)] #Kovarianz pro Objekt aus der gesamten P matrix extraieren 
                index_opt = indexes_opt[i][1]
                z_opt_assossiation = 0 #Initialisierung von : Messung der wahrscheinlichsten Hypothese
                if index_opt< m :
                    theta_k[0][i] = index_opt +1

                    if dimensions ==1 :
                       z_opt_assossiation = measurement_k[index_opt] # Messung der wahrscheinlichsten Hypothese
                    else:
                       z_opt_assossiation = measurement_k[index_opt]
                    
                
                else :
                    theta_k[0][i] = 0
                    
                

                estimate_i,P_i = kalman_filter_update(estimate_i,P_i,H,z_opt_assossiation,theta_k[0][i],R,dimensions) #Update P und estimate_i mit Kalman-Korrekturschritt
                
                P[0:number_states,i*number_states:number_states*(i+1)] = P_i #P_i in die gesamte P Matrix wieder einfügen
                estimate[0:number_states,i] = estimate_i #estimates_i in die gesamte estimates Matrix wieder einfügen
                
                

              
                
                
            estimate_all.append(estimate.tolist())
            weight_opt_k = np.exp(total_cost)
            k = k+1
            
            
            mn_data.pop(0) #Löschen ältestes Element
            mn_data.append(measurement_k) #Aktuelle Messung einfügen
            n = 2
            #n = mnLogic(M,N,1,mn_data) #Anzahl Objekte
         
        return estimate_all,n
            
     


