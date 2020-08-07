import  matplotlib.pyplot as plt
import numpy as np
import gnn_algorithmus
from functions import gaussian 
from functions import createTestDataSet
from functions import get_position
import math
from MN import mnLogic

#Initialisierung
p_d = 0.95 #Detektionsrate
lambda_c =0.3 #Clutter Intensität
V= 5 #Cluttering Volume
T= 0.1 #Abtastzeit
F = [[1,T],
     [0,1]] #Systemmatrix
H = [1,0] #Ausgangsmatrix 
Q = [[100 ,0],[0 ,100]] #Varianz des Modellrauschens
R = [[10]] #Varianz des Messrauschens
init_x1 = -1
init_x2 = 15 
init_values =np.array([[init_x1,init_x2],[0, 0]])
P_i_init = [[10,0],[0,10]]
M = 4 #Anzahl der benoetigten Detektionen
N= 5 #Anzahl der Scans

#Test Datensatz
warmup_data, measurements, real_objects,K = createTestDataSet()
#GNN Aufruf
data = measurements[:]
all_measurements = np.concatenate((warmup_data,measurements),axis=0).tolist()
n = mnLogic(M,N,1,all_measurements) #Anzahl Objekte
estimate_gnn = gnn_algorithmus.gnn(data,p_d,lambda_c,F,H,n,R,Q,init_values,P_i_init)
position_gnn = get_position(estimate_gnn)
print(position_gnn)

print('............................................')
print(real_objects)



#Plots


for i in K:
    for j in range(len(measurements[i])):
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
