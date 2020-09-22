import  matplotlib.pyplot as plt
import numpy as np
from gnn_algorithmus import *
from functions import *
import math
from MN import*
from ObjectHandler import *

#Initialisierung
dimensions = 2
p_d = 0.999 #Detektionsrate
T= 0.5 #Abtastzeit
M = 3 #Anzahl der benoetigten Detektionen
N= 4 #Anzahl der Scans
ObjectHandler = ObjectHandler()
Q = [[1000,0,0,0],
             [0,100,0,0],
             [0,0,100,0],
             [0,0,0,100]] #Varianz des Modellrauschens 2D
R = [[10,0],
    [0,10]] #Varianz des Messrauschens
P_i_init = [[100,0,0,0],[0,100,0,0],[0,0,100,0],[0,0,0,100]] 
treshhold = 0.06
#GNN Aufruf
estimate_gnn,n = gnn(p_d,M,N,dimensions,T,ObjectHandler,Q,R,P_i_init,treshhold)
#estimate_gnn,n = gnn_testdaten(p_d,M,N,dimensions,T)
#position_gnn = get_position(estimate_gnn,dimensions)






##Plots Testdaten
#_,measurements, real_objects,K = createTestDataSet(dimensions)
#if dimensions == 1:

#    for i in K:
#        for j in range(len(measurements[i])): #Plot Clutter
#            plt.plot(measurements[i][j],K[i]+1,'ro',color='black')
        
#        for j in range(n):
#            plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
#            plt.plot(position_gnn[i][j],K[i],'ro',color= 'orange')
#            plt.plot(position_gnn[-1][j],K[-1]+1,'ro',color= 'orange')
#    plt.legend('Z_k','x_ist')     
#    plt.title('Messungen')
#    plt.xlabel('x')
#    plt.ylabel('k')
#    plt.axis([-15,30,-1,len(K)+1])
#    plt.show()

#else:
#    fig, axs = plt.subplots(2)
#    fig.suptitle('Performance GNN')
#    for i in K:
#        for j in range(len(measurements[i])): #Plot Clutter
#            meas_k = measurements[i][j]
#            axs[0].plot(meas_k[0],K[i]+1,'ro',color='black')
#            axs[1].plot(meas_k[1],K[i]+1,'ro',color='black')
#        for j in range(n):
#            real_objects_ij = real_objects[i][j]
#            x_coordinates =  position_gnn[i][0]
#            y_coordinates = position_gnn[i][1]
#            x_end_position = position_gnn[-1][0]
#            y_end_position = position_gnn[-1][1]
#            axs[0].plot(real_objects_ij[0],K[i]+1,'ro',color='green') #x Koordinaten, echte Objekte
#            axs[1].plot(real_objects_ij[1],K[i]+1,'ro',color='green') #y Koordinaten, echte Objekte
#            axs[0].plot(x_coordinates[j],K[i],'ro',color= 'orange') #x Koordinaten, gnn
#            axs[1].plot(y_coordinates[j],K[i],'ro',color= 'orange') #y Koordinaten, gnn
#            axs[0].plot(x_end_position[j],K[-1]+1,'ro',color= 'orange') #letzter x punkt, gnn
#            axs[1].plot(y_end_position[j],K[-1]+1,'ro',color= 'orange') #letzter y punkt, gnn
#    axs[0].axis([0,21,-1,len(K)+1])
#    axs[1].axis([0,100,-1,len(K)+1])
#    axs[0].set_xlabel('x')
#    axs[0].set_ylabel('k')
#    axs[1].set_xlabel('y')
#    axs[1].set_ylabel('k')
#    plt.show()
