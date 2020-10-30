import  matplotlib.pyplot as plt

import random
import math
from typing import List
import time

from pyrate_common_math_gaussian import Gaussian
from pyrate_sense_filters_gmphd import GaussianMixturePHD
from plotGMM import *

import numpy as np
from numpy import vstack, array, ndarray, eye

def phd_BirthModels(obj_w, obj_h, num_w: int, num_h: int) -> List[Gaussian]:
    """
     Args:
            obj_w: number of pixels on width      
            obj_h: number of pixels on width
            num_w: number of gm on width
            num_h: number of gm on height     
    """
    birth_belief: List[Gaussian] = []
    b_leftside: List[Gaussian] = [] 
    b_rightside: List[Gaussian] = []
    b_area: List[Gaussian] = []
    # gm left side
    cov_edge = array([[15, 0.0,         0.0, 0.0], 
                     [0.0, obj_h/(4*num_h), 0.0, 0.0],
                     [0.0, 0.0,         5.0, 0.0],
                     [0.0, 0.0,         0.0, 5.0]])
    for i in range(1,num_h):
        mean = vstack([0, i*obj_h/(num_h+1), 1.0, 0.0])
        b_leftside.append(Gaussian(mean, cov_edge, weight=0.5))
    
    # gm right side
    for i in range(1,num_h):
        mean = vstack([obj_w, i*obj_h/(num_h+1), -1.0, 0.0])
        b_rightside.append(Gaussian(mean, cov_edge, weight=0.5))

    cov_edge = array([[100, 0.0,         0.0, 0.0], 
                     [0.0, 100, 0.0, 0.0],
                     [0.0, 0.0,         5.0, 0.0],
                     [0.0, 0.0,         0.0, 5.0]])
    mean = vstack([obj_w/2, 0+10, 0.0, 0.0])
    b_area.append(Gaussian(mean, cov_edge, weight=1))
    mean = vstack([obj_w/2, obj_h/2, 0.0, 0.0])
    b_area.append(Gaussian(mean, cov_edge, weight=1))
    mean = vstack([obj_w/2, obj_h-10, 0.0, 0.0])
    b_area.append(Gaussian(mean, cov_edge, weight=1))

    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)
    birth_belief.extend(b_area)

    return birth_belief

if __name__ == '__main__':
    # init
    F = array([[1.0, 0.0, 1.0, 0.0], 
            [0.0, 1.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]])
    H = array([[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]])
    Q = .1*eye(4)
    R = .05*eye(2)

    obj_h = 50
    obj_w = 50
    # create birth belief
    birth_belief = phd_BirthModels(50,50,10,10)
    # survival rate, detection rate and clutter intensity
    survival_rate = 0.99
    detection_rate = 0.9
    intensity = 0.01

    # phd object
    phd = GaussianMixturePHD(birth_belief,survival_rate,detection_rate,intensity,F,H,Q,R)

    objects: List[ndarray] = []
    meas: List[ndarray] = []
    meas_obj: List[ndarray] = []
    pos_phd: List[ndarray] = []

    obj_1_x_list: List[ndarray] = []
    obj_1_y_list: List[ndarray] = []
    obj_2_x_list: List[ndarray] = []
    obj_2_y_list: List[ndarray] = []
    obj_3_x_list: List[ndarray] = []
    obj_3_y_list: List[ndarray] = []
    obj_4_x_list: List[ndarray] = []
    obj_4_y_list: List[ndarray] = []

    # ==============
    # create objects
    print('creating objects...')
    random.seed(55)
    # good seeds: 30,55

    # first ten timesteps
    for i in range(10):
        obj_1_pos = array([[0.+i], [10.+i]])
        obj_2_pos = array([[50.-1.5*i], [15.]])
        obj_3_pos = array([[50.-i], [15.+i]])
        obj_4_pos = array([[25.+i/10], [10.+i]])
        obj_1_x_list.append(obj_1_pos[0])
        obj_1_y_list.append(obj_1_pos[1])
        obj_2_x_list.append(obj_2_pos[0])
        obj_2_y_list.append(obj_2_pos[1])
        obj_3_x_list.append(obj_3_pos[0])
        obj_3_y_list.append(obj_3_pos[1])
        obj_4_x_list.append(obj_4_pos[0])
        obj_4_y_list.append(obj_4_pos[1])
        objects.append([obj_1_pos, obj_2_pos, obj_3_pos, obj_4_pos])
    # timesteps 11 to 30
    for i in range(20):
        obj_1_pos = array([[10.+1.5*i], [20.+i]])
        obj_2_pos = array([[35.-1.5*i], [15.]])
        obj_3_pos = array([[40.-i], [26.+i]])
        obj_4_pos = array([[25.+(i+10)/10], [10.+(i+10)]])
        obj_1_x_list.append(obj_1_pos[0])
        obj_1_y_list.append(obj_1_pos[1])
        obj_2_x_list.append(obj_2_pos[0])
        obj_2_y_list.append(obj_2_pos[1])
        obj_3_x_list.append(obj_3_pos[0])
        obj_3_y_list.append(obj_3_pos[1])
        obj_4_x_list.append(obj_4_pos[0])
        obj_4_y_list.append(obj_4_pos[1])
        objects.append([obj_1_pos, obj_2_pos, obj_3_pos, obj_4_pos])
        
    # ======================
    # calculate measurements
    print('calculating measurements...')

    # degree of inaccuracy
    inac_intensity = 1.5
    # timestemps 1 to 10
    for i in range(10):
        inac_x = ((random.random()-0.5)*inac_intensity)
        inac_y = ((random.random()-0.5)*inac_intensity)
        obj_1_meas = array([obj_1_x_list[i]+inac_x, obj_1_y_list[i]+inac_y])
        obj_2_meas = array([obj_2_x_list[i]+inac_x, obj_2_y_list[i]+inac_y])
        obj_3_meas = array([obj_3_x_list[i]+inac_x, obj_3_y_list[i]+inac_y])
        obj_4_meas = array([obj_4_x_list[i]+inac_x, obj_4_y_list[i]+inac_y])
        meas_obj.append([obj_1_meas, obj_2_meas, obj_3_meas, obj_4_meas])
        meas.append([obj_1_meas, obj_2_meas, obj_3_meas, obj_4_meas, array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]) ] )
        
        # without measurement inaccuracies
        # meas.append([array([[0.+i], [10.+i]]), array([[50.-1.5*i], [15.]]), array([[50.-i], [15.+i]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]) ] )
    # timestemps 21 to 30
    for i in range(20):
        inac_x = ((random.random()-0.5)*inac_intensity)
        inac_y = ((random.random()-0.5)*inac_intensity)
        obj_1_meas = array([obj_1_x_list[i+10]+inac_x, obj_1_y_list[i+10]+inac_y])
        obj_2_meas = array([obj_2_x_list[i+10]+inac_x, obj_2_y_list[i+10]+inac_y])
        obj_3_meas = array([obj_3_x_list[i+10]+inac_x, obj_3_y_list[i+10]+inac_y])
        obj_4_meas = array([obj_4_x_list[i+10]+inac_x, obj_4_y_list[i+10]+inac_y])
        meas_obj.append([obj_1_meas, obj_2_meas, obj_3_meas, obj_4_meas])
        meas.append([obj_1_meas, obj_2_meas, obj_3_meas, obj_4_meas, array([[50*random.random()], [50*random.random()]]),  array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]), array([[50*random.random()], [50*random.random()]]) ] )

    # =================
    # plot birth belief
    """ fig = plt.figure()
    fig = plotGMM(gmm=birth_belief, pixel_w=obj_w, pixel_h=obj_h,figureTitle='Plot GMM Birth')
    plt.show() """

    # ==========
    # run GM-PHD
    print('starting gmphd...')
    pt = time.time()
    for ts,z in enumerate(meas, start=1):
        print('timestep %d' % ts)
        phd.predict()
        phd.correct(z)
        phd.prune(array([0.1]), array([3]), 20)
        pos_phd.append(phd.extract())
        
        # check min and max weight
        minWeight = 1
        maxWeight = 0
        for comp in phd.gmm:
            if comp.w > maxWeight:
                maxWeight = comp.w
        for comp in phd.gmm:
            if comp.w < minWeight:
                minWeight = comp.w
        print('min weight: %.3f | max weight: %.3f' % (minWeight, maxWeight))
        
        """ 
        # plot gaussian mixture model with measurements and extracted objects
        fig = plt.figure()
        fig = plotGMM(gmm=phd.gmm, pixel_w=obj_w, pixel_h=obj_h,figureTitle='Plot GMM Birth')
        for m in z:
            # measurements
            plt.plot(m[0], m[1], 'ro',color= 'white', ms= 2)
        for m in phd.extract():
            # extracted objects
            plt.plot(m[0],m[1],'ro',color= 'red', ms= 1)
        # plt.show() """
        
    print('finished filter in %.3f seconds' % (time.time()-pt))
    # =====
    # Plots

    K = np.arange(len(meas))

    # plot cardinality
    num_targets_truth = []
    num_targets_estimated = []

    for x_set in objects:
        num_targets_truth.append(len(x_set))
    for x_set in pos_phd:
        num_targets_estimated.append(len(x_set))

    plt.figure()
    (markerline, stemlines, baseline) = plt.stem(num_targets_estimated, label='Geschätze Anzahl Objekte')
    plt.setp(baseline, color='k')  # visible=False)
    plt.setp(stemlines, visible=False)  # visible=False)
    plt.setp(markerline, markersize=3.0)
    plt.step(num_targets_truth, 'r', label='tatsächliche Anzahl Objekte')
    plt.xlabel('k [Zeitschritte]')
    plt.legend()
    plt.title('Geschätzte Kardinalität VS tatsächliche Kardinalität', loc='center', wrap=True)


    # plot x-coordinates
    fig = plt.figure()
    # first plot trajectories
    plt_objects, = plt.plot(range(len(obj_1_x_list)),obj_1_x_list,color='red', label='Objekt Trajektorie')
    plt.plot(range(len(obj_2_x_list)),obj_2_x_list,color='red')
    plt.plot(range(len(obj_3_x_list)),obj_3_x_list,color='red')
    plt.plot(range(len(obj_4_x_list)),obj_4_x_list,color='red')
    for i in K:
        # measurements
        for j in range(len(meas[i])):
            plt_measurements, = plt.plot(K[i],meas[i][j][0],'bx', ms=4, label='Messungen')
        # object measurements
        for j in range(len(objects[i])):
            plt_object_measurements, = plt.plot(K[i],meas_obj[i][j][0],'k^',ms=5, label='Objekt Messung')
        # estimates
        for l in range(len(pos_phd[i])):
            plt_estimates, = plt.plot(K[i],pos_phd[i][l][0],'co', ms=3, label='Schätzungen')
            
    plt.title('x-Koordinaten')
    plt.xlabel('k [Zeitschritte]')
    plt.ylabel('x-Koordinate')
    plt.legend(handles=[plt_measurements, plt_estimates, plt_object_measurements, plt_objects], loc='best')
    plt.axis([-1,len(K)+1,-5,55])


    # plot y-coordinates
    fig = plt.figure()
    # first plot trajectories
    plt_objects, = plt.plot(range(len(obj_1_y_list)),obj_1_y_list,color='red', label='Objekt Trajektorie')
    plt.plot(range(len(obj_2_y_list)),obj_2_y_list,color='red')
    plt.plot(range(len(obj_3_y_list)),obj_3_y_list,color='red')
    plt.plot(range(len(obj_4_y_list)),obj_4_y_list,color='red')
    for i in K:
        # measurements
        for j in range(len(meas[i])):
            plt_measurements, = plt.plot(K[i],meas[i][j][1],'bx', ms=4, label='Messungen')
        # object measurements
        for j in range(len(meas_obj[i])):
            plt_object_measurements, = plt.plot(K[i],meas_obj[i][j][1],'k^',ms=5, label='Objekt Messung')
        # estimates
        for l in range(len(pos_phd[i])):
            plt_estimates, = plt.plot(K[i],pos_phd[i][l][1],'co', ms=3, label='Schätzungen')
            
    plt.title('y-Koordinaten')
    plt.xlabel('k [Zeitschritte]')
    plt.ylabel('y-Koordinate')
    plt.legend(handles=[plt_measurements, plt_estimates, plt_object_measurements, plt_objects], loc='upper left')
    plt.axis([-1,len(K)+1,-5,55])



    # plot x-y-view
    fig = plt.figure()
    # first plot trajectories
    plt_objects, = plt.plot(obj_1_x_list,obj_1_y_list,color='red', label='Objekt Trajektorie')
    plt.plot(obj_2_x_list,obj_2_y_list,color='red')
    plt.plot(obj_3_x_list,obj_3_y_list,color='red')
    plt.plot(obj_4_x_list,obj_4_y_list,color='red')

    for i in K:
        # measurements
        for j in range(len(meas[i])):
            plt_measurements, = plt.plot(meas[i][j][0],meas[i][j][1],'bx', ms=4, label='Messungen')

        # object measurements
        for j in range(len(meas_obj[i])):
            plt_object_measurements, = plt.plot(meas_obj[i][j][0],meas_obj[i][j][1],'k^',ms=5, label='Objekt Messung')

        # estimates
        for l in range(len(pos_phd[i])):
            plt_estimates, = plt.plot(pos_phd[i][l][0],pos_phd[i][l][1],'co', ms=3, label='Schätzungen')

    # plot settings
    plt.legend(handles=[plt_measurements, plt_estimates, plt_object_measurements, plt_objects], loc='upper left')
    plt.title('x-y-Ebene')
    plt.xlabel('x-Koord.')
    plt.ylabel('y-Koord.')
    plt.axis([0,50,0,50])
    plt.show()
