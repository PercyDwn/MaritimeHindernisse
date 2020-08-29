
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
H =[[1,0,0,0],
    [0,0,1,0]]#Ausgangsmatrix
H_velocity = [[0,1,0,0],
              [0,0,0,1]]
est = np.array([[1,10],[0.5,0.1],[2,8],[0.4,0.2]]) # Zustände (Eingang MN). Erste Spalte sind die Zustände des ersten Objekts. Die zweite des zweiten
#est = np.array([[1,0.5,10,0.1],[2,0.4,8,0.2]]) # Zustände (Eingang MN). Erste Spalte sind die Zustände des ersten Objekts. Die zweite des zweiten
vel_old = np.matmul(H_velocity,est).tolist() #Geschwindigkeit alt
pos_old = np.matmul(H,est).tolist() #Positionen alt (Eingangs des MN)
vel_new = [[0,0],[0,0],[0,0],[0,0]] #Geschwinigkeit neu. Erstmal als 0 annehmen. Später können wir dies erweitern 
#pos_new = np.array([[0.8,9.5],[1.5,8.2],[3,4],[10,12]]).tolist() #Neue Positionen aus dem M/N Algorithmus (List). Im Fall von Births
pos_new = np.array([[1.5,8.2]]).tolist() #Neue Positionen aus dem M/N Algorithmus (List). Im Fall von Deaths
n_new = len(pos_new) #neue Objektanzahl
est_updated = []
est_i = np.zeros((4)) #Zustände eines einezelnen Objekts. 4 ist die Anzahl der Zuständen

#Erweiterung : nur im Fall, dass Deaths oder Births auftreten n_alt != n_new.
if n_new > n_old: #Births: Die Koordinaten die die minimale Abstände von den alten Objekten aufweisen, werden als "schon vorhandetes Objekt" betrachtet und daher werden als neues Objekt ausgeschlossen
    for i in range(n_old):
        distances = [] #Liste mit Abständen zu den alten koordinaten
        for j in range(len(pos_new)):
            distances.append(differenz(pos_old[i],pos_new[j])) #Vektorbetrag hinzufügen
        index_min = min(range(len(distances)), key=distances.__getitem__) #minimalen index ausrechnen
        pos_new.pop(index_min)# Koordinaten mit kleinsten Abständen löschen 
        vel_new.pop(index_min)
        
         
    pos_updated = np.concatenate((pos_old,pos_new),axis=0)# Anfangswerte basierend auf bekannten und unbekannten Objekte
    vel_updated = np.concatenate((vel_old,vel_new),axis=0)
    for i in range(n_new):
        est_i[0] = pos_updated[i,0]
        est_i[1] = vel_updated[i,0]
        est_i[2] = pos_updated[i,1]
        est_i[3] = vel_updated[i,1]
        est_updated.append(est_i.tolist())
print('....') 
print(est)
print('....')
print(est_updated)
print('....')
if n_new < n_old: #Deaths: Die Koordinaten die die maximale Abstände von den alten Objekten aufweisen, werden als "sterbende Objekte" betrachtet und daher werden von den Zuständen gelöscht 
    for i in range(n_new):
        distances = [] #Liste mit Abständen zu den alten koordinaten
        for j in range(n_old):
            distances.append(differenz(pos_old[j],pos_new[i])) #Vektorbetrag hinzufügen
        index_min =  min(range(len(distances)), key=distances.__getitem__) #maximalen index ausrechnen
        a = pos_old[index_min][0]
        est_i[0] = pos_old[index_min][0]# Koordinaten mit kleinsten Abständen löschen 
        est_i[1] = vel_old[index_min][0]
        est_i[2] = pos_old[index_min][1]
        est_i[3] = vel_old[index_min][1]
        est_updated.append(est_i.tolist())
    print('....') 
    print(est)
    print('....')
    print(est_updated)
    print('....')
    a=1


  





