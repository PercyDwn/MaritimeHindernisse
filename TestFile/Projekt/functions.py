import numpy as np
import math
import matplotlib.pyplot as plt
import random 

def gaussian(x,my,var): #Funktion berechnet die Gaß sche Verteilung in abhängigkeit vom Wert, Erwartungswert und Varianz
    st_abw = np.sqrt(var) #standard Abweichung
    gaussian = 1/(2*math.pi)*np.exp(-0.5*((x-my)/st_abw)**2)
    return gaussian

def createTestDataSet(dimensions):
        warmup_data =  [[-1,0,6,13],
             [-1,3.5,4,14,15],
             [-5,-1.5,6,7, 14],
             [-2,4.5,8,13,15],
             ]
        
        if dimensions ==1:
           measurements = [[1,2.5,6,12],
                 [3,3.5,4,13,13.5],
                 [4.5,6,7, 14],
                 [1,4,4.5,8,13,15],
                 [4,4.5,5,6,16],
                 [4,4.5,16,17,17.5,18],
                 [1,2,3.5,3.6,15,15.5, 17,18]]
           objects = [[2.5,12],
             [3.5,13],
             [4.5, 14],
             [4.5 ,15],
             [4.5,16],
             [4,17],
             [3.5, 17]]
        else:
       
            measurements = [[[1,5],[2.5,10],[6,50],[12,70]],
                           [[3,60],[3.5,15],[4,40],[13,65],[13.5, 10]],
                           [[4.5,20],[6,70],[7,70],[14,60]],
                           [[1,10],[4.5,25],[8,35],[13,55],[15,55]],
                           [[4,0],[4.5,25],[5,35],[16,50]],
                           [[4,25],[5,50],[17,45],[17.5 ,25],[18,75]],
                           [[1,100],[3.5,30],[3.6,20],[15,10],[15.5,10],[17,40],[18,40]]
                           ]
            objects = [[[2.5,10], [12,70]],
                       [[3.5,15],[13,65]],
                       [[4.5,20],[14,60]],
                       [[4.5,25],[15,55]],
                       [[4.5,25],[16,50]],
                       [[4,25],[17,45]],
                       [[3.5,30],[17,40]]]
        K= np.arange(len(measurements))
        return warmup_data,measurements, objects,K

def erwartungs_wert():
        pass
def initialize_values(dimensions,T,n,measurements_0):
    #Test Datensatz
    m = len(measurements_0) #Anzahl der Messungen pro Zeitschritt
    init_values = np.zeros((2*dimensions,n)) + 1#Initialisierung der Anfagnswerten 
    #Anfangswertgenerator: Nimmt als Anfangswert einen zufälligen Wert aus dem ersten Zeitschritt der Messungen plus einen Abweichungsfaktor


    
    if dimensions ==1:
        F = [[1,T],
             [0,1]] #Systemmatrix
        H = [1,0] #Ausgangsmatrix 
        Q = [[100 ,0],[0 ,100]] #Varianz des Modellrauschens
        R = [[10]] #Varianz des Messrauschens
        #Anfangswertgenerator: Nimmt als Anfangswert einen zufälligen Wert aus dem ersten Zeitschritt der Messungen plus einen Abweichungsfaktor
       
        for i in range(n):
            random_meas_index = random.randint(0, m-1) #Zufälliger Wert zwischen 0 und m-1
            init_values[0,i] = measurements_0[random_meas_index]+  measurements_0[random_meas_index]/30
        P_i_init = [[10,0],[0,10]]
    else:
        F = [[1,T,0,0],
             [0,1,0,0],
             [0,0,1,T],
              [0,0,0,1]] #Systemmatrix 
        H =[[1,0,0,0],
            [0,0,1,0]]#Ausgangsmatrix
        Q = [[10,0,0,0],
             [0,10,0,0],
             [0,0,10,0],
             [0,0,0,10]] #Varianz des Modellrauschens 2D
        R = [[1,0],
             [0,1]] #Varianz des Messrauschens
        for i in range(n):
            random_meas_index = random.randint(0, m-1) #Zufälliger Wert zwischen 0 und m-1
            random_meas_coordinates = measurements_0[random_meas_index] #Zufällige Koordinaten aus der ersten Messung
            init_values[0,i] = random_meas_coordinates[0]+  random_meas_coordinates[0]/30 #x ELement aus der zufälligen Koordinate plus Abweichung
            init_values[2,i] = random_meas_coordinates[1]+  random_meas_coordinates[1]/30 #y ELement aus der zufälligen Koordinate plus Abweichung
        
        P_i_init = [[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]] 
        
    return F,H,Q,R, init_values,P_i_init



def get_position(estimate,dimensions): #Extraiert die Position der Objekten aus der Zustandsschätzung
    
    if dimensions ==1:
        position = [None] * len(estimate)
        for i in range(len(estimate)):
            position[i] = estimate[i][0]
    else:
        position = estimate[:]
        for j in range(len(estimate)):
            estimate_k = estimate[j]
            position_k = [None]*int(len(estimate_k)/2)
            for i in range(int(len(estimate_k)/2)):
                position_k[i] = estimate_k[2*i]
            position[j] = position_k
            
    return position