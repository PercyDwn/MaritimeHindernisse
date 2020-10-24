
from ObjectHandler import *

import  matplotlib.pyplot as plt
import numpy as np
import pyrate_sense_filters_gmphd
import random
from pyrate_common_math_gaussian import Gaussian
from pyrate_sense_filters_gmphd import GaussianMixturePHD
from numpy import vstack
from numpy import array
from numpy import ndarray
from numpy import eye
import math
from mpl_toolkits.mplot3d import Axes3D
import cv2

from phd_plot import *
from plotGMM import *
from phd_functions import * 

# Typing
from typing import List, Tuple, cast

# ObjectHandler init
ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1_0')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)
ObjectHandler.setPlotOnUpdate(True)

# GM_PHD filter init
F = array([[1.0, 0.0, 1.0, 0.0], 
           [0.0, 1.0, 0.0, 1.0], 
           [0.0, 0.0, 1.0, 0.0], 
           [0.0, 0.0, 0.0, 1.0]])
H = array([[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0]])
Q = 20*eye(4)
R = array(
    [[7, 0],
     [0, 50]])

HorizontLines = ObjectHandler.getHorizonLines(t=1)
ypt = 0
for hl in HorizontLines:
    ypt = max(ypt, hl.height)


# image size
img_w = 640
img_h = 480
# create birth_belief
boatContour = ('Triangle', 30,120)
ypt = cast(int, ypt) - 10
birth_belief = phd_BirthModels(obj_w = img_w, obj_h = img_h, num_w = 8, num_h = 6, ypt = ypt, xpe = 0, boatContour=boatContour)
# plot gmm
imgPath = os.getcwd()+'/TestFile/Projekt/Bilder/list1/1.jpg'
fig = plotGMM(gmm = birth_belief, pixel_w = img_w, pixel_h = img_h, detail = 1 , method = 'rowwise', figureTitle = 'Birth Belief GMM', savePath = '/PHD_Plots', imgPath = imgPath)
fig.show()

survival_rate = 0.999
detection_rate = 0.9
intensity = 0.0001

# phd object
phd = GaussianMixturePHD(birth_belief, survival_rate, detection_rate, intensity, F, H, Q, R)

# measurements
print('get measurements ...')
meas: List[ndarray] = []
ObjectHandler.setPlotOnUpdate(True)
for k in range(1,21):
    meas.insert(k,  ObjectHandler.getObjectStates(k, 'cb'))

print('rearrange measurements ...')
meas_v: List[ndarray] = []
for k in range(len(meas)):
    meas_vk: ndarray = []

    for j in range(len(meas[k])):
        meas_vk.append(array([[meas[k][j][0]], [meas[k][j][1]]]))
    meas_v.insert(k, meas_vk)

# apply phd filter
print('start phd filtering...')
pos_phd: List[ndarray] = []
ci = 1
for z in meas_v:
    phd.predict()
    phd.correct(z)
    fig = plotGMM(gmm = phd.gmm, pixel_w = 640, pixel_h = 480, detail = 1 , method = 'rowwise', figureTitle = 'PHD GMM k-' + str(ci)+'before pruning', savePath = '/PHD_Plots')
    phd.prune(array([0.01]), array([20]), 100)
    pos_phd.append(phd.extract())
    print( 'timestep ' + str(ci) )
    print( 'tracking ' + str(len(phd.extract())) + ' objects' )
    fig = plotGMM(gmm = phd.gmm, pixel_w = 640, pixel_h = 480, detail = 1 , method = 'rowwise', figureTitle = 'PHD GMM k-' + str(ci)+'after pruning', savePath = '/PHD_Plots')
    for l in range(len(z)):
       plt.plot(z[0],z[1],'ro',color= 'black', ms= 1)
    for est in phd.extract():
       plt.plot(est[0],est[1],'ro',color= 'red', ms= 1)
    #plt.show()
    fig.close()
    ci += 1
    print('------------------')
    
# plot data

# xy 
for i in range(len(meas_v)):
    #Messungen
    for j in range(len(meas_v[i])):
        plt.plot(meas_v[i][j][0],meas_v[i][j][1],'ro',color='black')

    #Schätzungen
    for l in range(len(pos_phd[i])):
        plt.plot(pos_phd[i][l][0],pos_phd[i][l][1],'ro',color= 'red', ms= 3)
        
#plt.legend(['Zk', 'phd'])     
plt.title('x-y-Raum')
plt.xlabel('x-Koord.')
plt.ylabel('y-Koord.')
plt.gca().invert_yaxis()
plt.show()

## x-axis
#K = np.arange(len(meas_v))

#for i in K:
#    #Messungen
#    for j in range(len(meas_v[i])):
#        plt.plot(K[i], meas_v[i][j][0],'ro',color='black')

#    #Schätzungen
#    for l in range(len(pos_phd[i])):
#        plt.plot(K[i], pos_phd[i][l][0],'ro',color= 'red', ms= 3)
        
##plt.legend(['Zk', 'phd'])     
#plt.title('x-Raum')
#plt.xlabel('zeitpunkt k')
#plt.ylabel('x-Koord.')
#plt.axis([-1,20,-5,645])
#plt.show()

## y-axis
#for k in K:
#    #Messungen
#    for j in range(len(meas_v[k])):
#        plt.plot(K[k], meas_v[k][j][1],'ro',color='black')

#    #Schätzungen
#    for l in range(len(pos_phd[k])):
#        plt.plot(K[k], pos_phd[k][l][1],'ro',color= 'red', ms= 3)
        
##plt.legend(['Zk', 'phd'])     
#plt.title('y-Raum')
#plt.xlabel('zeitpunkt k')
#plt.ylabel('y-Koord.')
#plt.axis([-1,20,-5,485])
#plt.show()

# Scater plot 3D
#------------------------------------------------------------------------
#fig = plt.figure()
#ax = Axes3D(fig)

#for i in K:
#    for j in range(len(meas_v[i])):
#        ax.scatter(meas_v[i][j][0],meas_v[i][j][1],K[i])

#ax.set_xlabel('X Axis')
#ax.set_ylabel('Y Axis')
#ax.set_zlabel('k')
#f()


#------------------------------------------------------------------------
# Plott DATA in Image
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#for i in range(1,20):
#    ObjectHandler.updateObjectStates(True)
#    cv2.drawMarker(img, obst.bottom_center, (0, 255, 0), cv2.MARKER_CROSS, 10, thickness=2)

#for k in range(1,20):
#    plot_PHD_realpic(ObjectHandler, pos_phd, k)



#number_states = len(F) # Zuständezahl
#k = 0   #Zeitschritt
#pictures_availiable = True
#fig = plt.figure()

#for i in range(1,20):
#    print('---------------------------------------')
#    success = ObjectHandler.updateObjectStates()
#    if success == True:
#        print('updated states for time step ' + str(i))
#    else:
#        print('could not update states for time step ' + str(i))
#    print('last object states:')
#    print(ObjectHandler.getLastObjectStates())
#    #cv2.waitKey(1000)

#print('---------------------------------------')
##print('current time step: ' + str(ObjectHandler.getTimeStepCount()))
##print(ObjectHandler.getLastObjectStates())

#for i in range(1,20):
#    try:
#        print('get data for timestep ' + str(i) + ':')
    
#        print(ObjectHandler.getObjectStates(i))
#    except InvalidTimeStepError as e:
#        print(e.args[0])
#    print('---------------------------------------')

#cv2.waitKey(0)