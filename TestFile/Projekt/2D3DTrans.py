import  matplotlib.pyplot as plt
# Mathematics
from numpy.linalg import inv
import numpy as np
from numpy import array
import math
from cv2 import Rodrigues

from scipy.spatial.transform import Rotation as R

def trans(u, v, fx, fy, cx, cy, R_quat: tuple, t) -> array:
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
    # create Rotation object with quatrions
    r = R.from_quat([R_quat[0], R_quat[1], R_quat[2], R_quat[3]])

    Rt = np.hstack(r.as_matrix(), t)

    Vec = inv(A)@inv(Rt)@v_pix

    return Vec