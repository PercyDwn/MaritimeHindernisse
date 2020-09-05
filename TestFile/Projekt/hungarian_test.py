
import numpy as np
from numpy import concatenate
import math
from numpy.linalg import multi_dot


def betrag(vektor):
    sum=0
    for i in range(len(vektor)):
        sum=sum+vektor[i]*vektor[i]
    return math.sqrt(sum)

def differenz(vektorA, vektorB):
    vektorC=[]
    for i in range(len(vektorA)):
        vektorC.append(0)
        vektorC[i]=vektorA[i]-vektorB[i]
    return betrag(vektorC)


n_old = 2 #Alte Objekt Anzahl
H= np.array([[1,0,0,0],[0,0,1,0]]) #Ausgangsmatrix
H_velocity = np.array([[0,1,0,0],[0,0,0,1]])
est = np.array([[1,2],[0.5,0.1],[10,8],[0.4,0.2]]) # Zustände (Eingang MN). Erste Spalte sind die Zustände des ersten Objekts. Die zweite des zweiten
vel_old = np.matmul(H_velocity,est) #Geschwindigkeit alt
pos_old = np.matmul(H,est) #Positionen alt (Eingangs des MN)
vel_new = np.array([[0,0,0,0],[0,0,0,0]]) #Geschwinigkeit neu. Erstmal als 0 annehmen. Später können wir dies erweitern 
#pos_new = np.array([[0.8,1.5,3,4],[9.5,8.2,10,12]]) #Neue Positionen aus dem M/N Algorithmus (List). Im Fall von Births  
pos_new = np.array([[1.5],[8.5]]) #Neue Positionen aus dem M/N Algorithmus (List). Im Fall von Deaths
n_new = pos_new.shape[1] #neue Objektanzahl
est_updated = np.zeros((4,n_new))
est_i = np.zeros((4)) #Zustände eines einezelnen Objekts. 4 ist die Anzahl der Zuständen

#Erweiterung : nur im Fall, dass Deaths oder Births auftreten n_alt != n_new.
if n_new > n_old: #Births: Die Koordinaten die die minimale Abstände von den alten Objekten aufweisen, werden als "schon vorhandetes Objekt" betrachtet und daher werden als neues Objekt ausgeschlossen
    for i in range(n_old):
        distances = [] #Liste mit Abständen zu den alten koordinaten
        for j in range(pos_new.shape[1]):
            distances.append(differenz(pos_old[:,i],pos_new[:,j])) #Vektorbetrag hinzufügen
        index_min = min(range(len(distances)), key=distances.__getitem__) #minimalen index ausrechnen
        pos_new = np.delete(pos_new,index_min,1)
        vel_new = np.delete(vel_new,index_min,1)
   
    est_updated[0,:] = np.concatenate((pos_old[0],pos_new[0]),0)
    est_updated[1,:] = np.concatenate((vel_old[0],vel_new[0]),0)
    est_updated[2,:] = np.concatenate((pos_old[1],pos_new[1]),0)
    est_updated[3,:] = np.concatenate((vel_old[1],vel_new[1]),0)
    
    
if n_new < n_old: #Deaths: Die Koordinaten die die maximale Abstände von den alten Objekten aufweisen, werden als "sterbende Objekte" betrachtet und daher werden von den Zuständen gelöscht 
    for i in range(n_new):
        distances = [] #Liste mit Abständen zu den alten koordinaten
        for j in range(n_old):
            distances.append(differenz(pos_old[:,j],pos_new[:,i])) #Vektorbetrag hinzufügen
        index_min =  min(range(len(distances)), key=distances.__getitem__) #maximalen index ausrechnen
        est_updated[0] = pos_old[0,index_min]
        est_updated[1] = vel_old[0,index_min]
        est_updated[2] = pos_old[1,index_min]
        est_updated[3] = vel_old[1,index_min]
    print(est_updated)
   
    


  





