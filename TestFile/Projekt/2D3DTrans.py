import  matplotlib.pyplot as plt
# Mathematics
from numpy.linalg import inv
import numpy as np
from numpy import array
import math




def trans(u, v, fx, fy, cx, cy, R, t) -> array:
    """
    Args:
        u: x Coordinates in Pixel
        v: y Coordinates in Pixel
        A: Camera Martrix / intrinsic matrix
        fx, fy: focal length in pixel units
        cx, cy: 
        R: rotian Matrix of the camera
        t: Translation Matrix of the camera
        [R|t]: joint rotation-translation matrix
       
    """
    v_pix = [[u],
             [v],
             [1]]
    A = [[fx, 0,  cx], 
         [0,  fy, cy], 
         [0,  0,  1]]
    Rt = np.hstack(R, t)

    Vec = inv(A)@inv(Rt)@v_pix

    return Vec