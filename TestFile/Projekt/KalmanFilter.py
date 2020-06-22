class KalmanFilter(object):
  """description of class"""

import  matplotlib.pyplot as plt
import numpy as np
import math



    #Systembeschreibung:
    # x_kp = A*x_k + B*u_k + n_x,k      Q=Cov(n_x)
    # y_k = C*x_k + n_y,k               R=Cov(n_y)

    def prediction(xhat_k, Phat_k, A, B, u_k, Q): #wie kann man fkt mit variablen Eignagnsmparam definieren? -> wenn kein B bzw u_k vorhanden
        xhat_kp=np.matmul(A, xhat) + np.matmul(B, u_k) #Prädiziertes x(k+1)
        Phat_kp=np.matmul(A, np.maltmul(Phat_k, np.transpose(A))) + Q #Prädiziertes P(k+1)
        return xhat_kp, Phat_kp   

    def update(xhat_k, Phat_k,y_k , C, R):
        K_k = (P_hatk.dot(np.transpose(C)).dot(np.linalg.inv(C.dot(Phat_k).dot(C.transpose())
        x_tilk = x_hatk + K_k.dot(y_k - C.dot(x_hatk)) #Korriegiertes x(k)
        P_tilk = Phat_k - K_k.dot(C).dot(Phat_k)       #Korrigertes P(k)
        return x_tilk, P_tilk