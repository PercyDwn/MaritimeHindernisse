import  matplotlib.pyplot as plt
import numpy as np
from gnn_algorithmus import *
from functions import *
import math
from MN import*
from ObjectHandler import *

#Initialisierung
dimensions = 2
p_d = 0.9999 #Detektionsrate
T= 0.5 #Abtastzeit
M = 3 #Anzahl der benoetigten Detektionen (M/N Algorithmus)
N= 4 #Anzahl der Scans vor der Ausgabe des Zustands (Zeitverzögerung)
ObjectHandler = ObjectHandler()
Q = [[100,0,0,0],
             [0,100,0,0],
             [0,0,100,0],
             [0,0,0,100]] #Varianz des Modellrauschens 2D
R = [[10,0],
    [0,10]] #Varianz des Messrauschens
P_i_init = [[100,0,0,0],[0,1,0,0],[0,0,100,0],[0,0,0,1]] #Initialisierung der Kovarianzmatrix des Scätzfehlers
treshhold = 0.08 #M/N Trashhold
Q_horizon = [[0.1,0],
            [0,0.1]] ##Varianz des Modellrauschens Horizont
R_horizon = 1 #Messvarianz des Horizonts
P_horizon = [[1,0],
            [0,1]]  #Anfangsvarianz des Scätzfehlers des Horizonts
#GNN Aufruf
estimate_gnn,n = gnn(p_d,M,N,dimensions,T,ObjectHandler,Q,R,P_i_init,treshhold,Q_horizon,R_horizon,P_horizon) #Aufruf des GNN











#estimate_gnn,n = gnn_testdaten(p_d,M,N,dimensions,T)
#position_gnn = get_position(estimate_gnn,dimensions)






##plots testdaten
#_,measurements, real_objects,k = createTestDataSet(dimensions)
#if dimensions == 1:

#    for i in k:
#        for j in range(len(measurements[i])): #plot clutter
#            plt.plot(measurements[i][j],k[i]+1,'ro',color='black')
        
#        for j in range(n):
#            plt.plot(real_objects[i][j],k[i]+1,'ro',color='green')
#            plt.plot(position_gnn[i][j],k[i],'ro',color= 'orange')
#            plt.plot(position_gnn[-1][j],k[-1]+1,'ro',color= 'orange')
#    plt.legend('z_k','x_ist')     
#    plt.title('messungen')
#    plt.xlabel('x')
#    plt.ylabel('k')
#    plt.axis([-15,30,-1,len(k)+1])
#    plt.show()

#else:
#    fig, axs = plt.subplots(2)
#    fig.suptitle('performance gnn')
#    for i in k:
#        for j in range(len(measurements[i])): #plot clutter
#            meas_k = measurements[i][j]
#            axs[0].plot(meas_k[0],k[i]+1,'ro',color='black')
#            axs[1].plot(meas_k[1],k[i]+1,'ro',color='black')
#        for j in range(n):
#            real_objects_ij = real_objects[i][j]
#            x_coordinates =  position_gnn[i][0]
#            y_coordinates = position_gnn[i][1]
#            x_end_position = position_gnn[-1][0]
#            y_end_position = position_gnn[-1][1]
#            axs[0].plot(real_objects_ij[0],k[i]+1,'ro',color='green') #x koordinaten, echte objekte
#            axs[1].plot(real_objects_ij[1],k[i]+1,'ro',color='green') #y koordinaten, echte objekte
#            axs[0].plot(x_coordinates[j],k[i],'r+') #x koordinaten, gnn
#            axs[1].plot(y_coordinates[j],k[i],'r+') #y koordinaten, gnn
#            axs[0].plot(x_end_position[j],k[-1]+1,'r+') #letzter x punkt, gnn
#            axs[1].plot(y_end_position[j],k[-1]+1,'r+') #letzter y punkt, gnn
#    axs[0].axis([0,21,-1,len(k)+1])
#    axs[1].axis([0,100,-1,len(k)+1])
#    axs[0].set_xlabel('x-koordinate')
#    axs[0].set_ylabel('zeitschritt k')
#    axs[1].set_xlabel('y-koordinate')
#    axs[1].set_ylabel('zeitschritt k')
#    axs[0].grid()
#    axs[1].grid()
#    plt.show()

##x,y koordinaten
#fig= plt.subplots(1)
#for i in k:
#        for j in range(len(measurements[i])): #plot clutter
#            meas_k = measurements[i][j]
#            plt.plot(meas_k[0],meas_k[1],'ro',color='black')
        
#        for j in range(n):
#            real_objects_ij = real_objects[i][j]
#            x_coordinates =  position_gnn[i][0]
#            y_coordinates = position_gnn[i][1]
#            x_end_position = position_gnn[-1][0]
#            y_end_position = position_gnn[-1][1]
#            plt.plot(real_objects_ij[0],real_objects_ij[1],'ro',color='green')
#            plt.plot(x_coordinates[j],y_coordinates[j],'r+')
#            #plt.plot(position_gnn[-1][j],k[-1]+1,'ro',color= 'orange')
#plt.legend('z_k','x_ist')     
#plt.title('ist-soll verglich scätzung der koorinaten')
#plt.xlabel('x koordinate')
#plt.ylabel('y koordinate')
##plt.axis([-15,30,-1,len(k)+1])
#plt.grid()
#plt.show()