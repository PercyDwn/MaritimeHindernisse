import  matplotlib.pyplot as plt
import numpy as np
import os

def plot_GNN(pos_k,meas_k,fig, axs,k,ObjectHandler):
    
    for m in range(len(meas_k)):
        axs[0].plot(meas_k[m][0],k,'ro',color='black')
        axs[1].plot(meas_k[m][1],k,'ro',color='black')
        absolute_value = (meas_k[m][0]**2+meas_k[m][1]**2)**0.5
        axs[2].plot(absolute_value, k,'ro',color='black')
    try:
        n = len(pos_k[0])
        for i in range(n):
             axs[0].plot(pos_k[0,i],k,'r+')
             axs[1].plot(pos_k[1,i],k,'r+')
             absolute_value_gnn = (pos_k[0,i]**2+pos_k[1,i]**2)**0.5
             axs[2].plot(absolute_value_gnn,k,'r+')
    except:
        print('No Object detected')
         

def plot_GNN_realpic(ObjectHandler,pos_k,k,N, real_pic,meas_k,height_hor):

    cwd = os.getcwd() 
    pfad = cwd +'\ObjektDetektion' #Aktueller pfad
    if k == N: #Bei dem ersten Zeitscchritt indem neue Bilder gespeichert werden, alle Bilder im Ordner löschen
        filelist = [ f for f in os.listdir(pfad) if f.endswith(".png") ]
        for f in filelist:
            os.remove(os.path.join(pfad, f))   
    name = '\Bild_'+ str(k)+'.png'
    img = ObjectHandler.getImg()
    img_h, img_w,_  = img.shape
    obj_h = ObjectHandler.getImgHeight()
    obj_w = ObjectHandler.getImgWidth()
    hor_point_x = [0,img_w]
    hor_point_y = [height_hor*img_h/obj_h, height_hor*img_h/obj_h]
    plt.imshow(img)
    
    #Plot aller Messungen
    for m in range(len(meas_k)):
        
        plt.plot(meas_k[m][0]*img_w/obj_w,meas_k[m][1]*img_h/obj_h,'ro',color='black')
    #Plot Objektdetektion
    try:
        n = len(pos_k[0])
        for i in range(n):
            plt.plot(pos_k[0,i]*img_w/obj_w,pos_k[1,i]*img_h/obj_h,'r+')
    except:
        print('Consider increasing the detection treshhold')

    
    plt.plot(hor_point_x, hor_point_y)
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
        
    
