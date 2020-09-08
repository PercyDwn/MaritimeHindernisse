import  matplotlib.pyplot as plt
import numpy as np
import os

####################################
# Überarbeiten/Anpassen auf phd_filter
####################################

def plot_GNN(pos_k,meas_k,fig, axs,k,ObjectHandler):
    n = len(pos_k[0])
    for m in range(len(meas_k)):
        axs[0].plot(meas_k[m][0],k,'ro',color='black')
        axs[1].plot(meas_k[m][1],k,'ro',color='black')
        absolute_value = (meas_k[m][0]**2+meas_k[m][1]**2)**0.5
        axs[2].plot(absolute_value, k,'ro',color='black')
    
    for i in range(n):
         axs[0].plot(pos_k[0,i],k,'r+')
         axs[1].plot(pos_k[1,i],k,'r+')
         absolute_value_gnn = (pos_k[0,i]**2+pos_k[1,i]**2)**0.5
         axs[2].plot(absolute_value_gnn,k,'r+')
         

def plot_PHD_realpic(ObjectHandler,pos_phd,k):

    cwd = os.getcwd()
    pfad = cwd +'\ObjektDetektion'
   
    name = '\Bild_'+ str(k)+'.png'
    real_pic = plt.figure()
    img = ObjectHandler.getImg()
    plt.imshow(img)
    n = len(pos_phd[k-1])
    for i in range(n):
        plt.plot(pos_phd[k-1][i][0],pos_phd[k-1][i][1],'r+')
    real_pic.savefig(pfad + name)
    

    
    
    
    
    
    
def plt_GNN_settings(fig,axs):
        fig.suptitle('Performance GNN')
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('k')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('k')
        axs[2].set_xlabel('Betrag')
        axs[2].set_ylabel('k')
        plt.show()
        
    

