import numpy as np
import math
import matplotlib.pyplot as plt

def gaussian(x,my,var): #Funktion berechnet die Gaß sche Verteilung in abhängigkeit vom Wert, Erwartungswert und Varianz
    st_abw = np.sqrt(var) #standard Abweichung
    gaussian = 1/(2*math.pi)*np.exp(-0.5*((x-my)/st_abw)**2)
    return gaussian

