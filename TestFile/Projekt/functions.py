import numpy as np
import math
import matplotlib.pyplot as plt

def gaussian(x,my,var): #Funktion berechnet die Gaß sche Verteilung in abhängigkeit vom Wert, Erwartungswert und Varianz
    st_abw = np.sqrt(var) #standard Abweichung
    gaussian = 1/(2*math.pi)*np.exp(-0.5*((x-my)/st_abw)**2)
    return gaussian

def createTestDataSet():
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
        K= np.arange(len(measurements))
        return measurements, objects,K

def erwartungs_wert():
        pass

def get_position(estimate):
    position = [None] * len(estimate)
    for i in range(len(estimate)):
        position[i] = estimate[i][0]
    return position