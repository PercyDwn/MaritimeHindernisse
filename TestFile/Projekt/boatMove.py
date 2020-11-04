# Typing
from typing import List

# Mathematics
import numpy as np
from numpy import ndarray

"""
Habe anfangs und Endpunkte 
bewegung ist linear mit const velo
Fps ist 5

=> Bewegungsmuster für Boote
"""

def BoatPos(start, end, fps, kmax, shift: float = 0.0) -> List[ndarray]:
    """
    Args:
        start: Startpunkt x,y,z Koordinaten in m
        end: Endpunkt x,y,z Koordinaten in m
        fps: Frames per Second
        kmax: Framezahl am Endpunkt
    """
    BoatPos: List[ndarray] = []
    delta_v = np.subtract(end,start)/(kmax-1)
    alpha = np.arctan(delta_v[1]/delta_v[0])
    #delta_v = (end-start)/kmax
    for k in range(kmax):
        pos_x = start[0]+(k*delta_v[0]) + (np.cos(alpha) * shift)
        pos_y = start[1]+(k*delta_v[1]) + (np.sin(alpha) * shift)
        BoatPos.append([ [pos_x],[pos_y] ])
    return BoatPos

  
