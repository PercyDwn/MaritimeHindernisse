from pyrate_common_math_gaussian import Gaussian

import numpy as np
from numpy import vstack
from numpy import array
from numpy import ndarray
from numpy import eye
import math

from typing import List, Tuple

def phd_BirthModelsOld (num_w: int, num_h: int) -> List[Gaussian]:
    """
     Args:
            ObjectHandler: ObjectHandler          
            num_w: number of fields on width
            num_h: number of Fields on height
            
    """

    

    obj_h = 50
    obj_w = 50

    birth_belief: List[Gaussian] = []

    # Birthmodelle Rand links
    #--------------------------
    b_leftside: List[Gaussian] = [] 
    cov_edge = array([[15, 0.0,         0.0, 0.0], 
                     [0.0, obj_h/(4*num_h), 0.0, 0.0],
                     [0.0, 0.0,         5.0, 0.0],
                     [0.0, 0.0,         0.0, 5.0]])
    for i in range(1,num_h):
        mean = vstack([0, i*obj_h/(num_h+1), 1.0, 1.0])
        b_leftside.append(Gaussian(mean, cov_edge))
    
    # Birthmodelle Rand rechts
    #--------------------------
    b_rightside: List[Gaussian] = [] 
    for i in range(1,num_h):
        mean = vstack([obj_w, i*obj_h/(num_h+1), -1.0, 1.0])
        b_rightside.append(Gaussian(mean, cov_edge))

    birth_belief.extend(b_leftside)
    birth_belief.extend(b_rightside)

    return birth_belief

def phd_BirthModels(obj_w: int, obj_h: int, num_w: int, num_h: int, ypt: int = 0, xpe: int = 20, boatContour: Tuple = None) -> List[Gaussian]:

    birth_belief: List[Gaussian] = []
    b_leftside: List[Gaussian] = [] 
    b_rightside: List[Gaussian] = [] 

    cov_mul = 10
    padding_edge = xpe
    cov_padding_edge = 0


    if boatContour and boatContour[0] == 'Triangle':
        margin = boatContour[1]
        height_offset = boatContour[2]
        x1=(margin)-1
        y1=(obj_h)+1
        x2=(obj_w - margin)+1
        y2=(obj_h)+1
        x3=(obj_w/2)
        y3=(obj_h-height_offset)-1
        checkTriangle = True
    else:
        checkTriangle = False

    cov_edge = array(
        [[(obj_w/num_w)+cov_padding_edge, 0.0,              0.0,    0.0], 
         [0.0,              (obj_h/num_h)+cov_padding_edge, 0.0,    0.0],
         [0.0,              0.0,              20.0,   0.0],
         [0.0,              0.0,              0.0,    20.0]])
    for i in range(0,num_h+1):
        mean_left = vstack([padding_edge, (i*(obj_h-2*ypt)/num_h)+ypt, 10.0, 0])
        b_leftside.append(Gaussian(mean_left, cov_mul*cov_edge))
        mean_right = vstack([obj_w-padding_edge, (i*(obj_h-ypt)/num_h)+ypt, -10.0, 0])
        b_rightside.append(Gaussian(mean_right, cov_mul*cov_edge))

    cov_area = array(
        [[obj_w/num_w,  0.0,            0.0,    0.0], 
         [0.0,          obj_h/num_h,    0.0,    0.0],
         [0.0,          0.0,            20.0,   0.0],
         [0.0,          0.0,            0.0,    20.0]])
    b_area: List[Gaussian] = []
    for i in range(0,num_h+1):
        for j in range(1, num_w): 
            xm=j*obj_w/num_w
            ym=(i*(obj_h-ypt)/num_h)+ypt
            if checkTriangle and pointInTriangle(x1,y1,x2,y2,x3,y3,xm,ym): continue
            mean = vstack([xm, ym, 0.0, 0.0])
            b_area.append(Gaussian(mean, cov_mul*cov_area))
    
    birth_belief.extend(b_leftside)
    birth_belief.extend(b_area)
    birth_belief.extend(b_rightside)

    return birth_belief
    
def pointInTriangle(x1:int ,y1:int ,x2:int ,y2:int ,x3:int ,y3:int ,xp:int ,yp:int ) -> bool:
    c1 = (x2-x1)*(yp-y1)-(y2-y1)*(xp-x1)
    c2 = (x3-x2)*(yp-y2)-(y3-y2)*(xp-x2)
    c3 = (x1-x3)*(yp-y3)-(y1-y3)*(xp-x3)
    if (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0): return True
    return False