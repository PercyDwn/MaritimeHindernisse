import  matplotlib.pyplot as plt
import numpy as np
import gnn_algorithmus
from functions import gaussian 
from functions import createTestDataSet
from functions import get_position
from functions import initialize_values
import math
from MN import mnLogic

#Initialisierung
dimensions = 2
p_d = 0.9999 #Detektionsrate
T= 0.1 #Abtastzeit
M = 4 #Anzahl der benoetigten Detektionen
N= 5 #Anzahl der Scans

warmup_data, measurements,real_objects,K = createTestDataSet(dimensions) #Testdaten
data = measurements[:]
all_measurements = np.concatenate((warmup_data,measurements),axis=0).tolist()
#n = mnLogic(M,N,1,all_measurements) #Anzahl Objekte
n = 2 
F,H,Q,R, init_values,P_i_init = initialize_values(dimensions,T,n,measurements[0]) #Initialisierung aller Anfangswerten 
#GNN Aufruf
estimate_gnn = gnn_algorithmus.gnn(data,p_d,F,H,n,R,Q,init_values,P_i_init)
#print(estimate_gnn)
position_gnn = get_position(estimate_gnn)
#print(position_gnn)

#print('............................................')
#print(real_objects)



#Plots
if dimensions ==1:

    for i in K:
        for j in range(len(measurements[i])): #Plot Clutter
            plt.plot(measurements[i][j],K[i]+1,'ro',color='black')
        
        for j in range(n):
            plt.plot(real_objects[i][j],K[i]+1,'ro',color='green')
            plt.plot(position_gnn[i][j],K[i],'ro',color= 'orange')
            plt.plot(position_gnn[-1][j],K[-1]+1,'ro',color= 'orange')
    plt.legend('Z_k','x_ist')     
    plt.title('Messungen')
    plt.xlabel('x')
    plt.ylabel('k')
    plt.axis([-15,30,-1,len(K)+1])
    plt.show()

else:
    fig, axs = plt.subplots(2)
    fig.suptitle('Performance GNN')
    for i in K:
        for j in range(len(measurements[i])): #Plot Clutter
            print(measurements)
            print(measurements[i])
            meas_k = measurements[i][j]
            print(meas_k)
            axs[0].plot(meas_k[0],K[i]+1,'ro',color='black')
            axs[1].plot(meas_k[1],K[i]+1,'ro',color='black')
    axs[0].axis([-15,30,-1,len(K)+1])
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('k')
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('k')
    plt.show()
