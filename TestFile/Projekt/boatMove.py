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

def BoatPos(start, end, fps, kmax) -> List[ndarray]:
    """
    Args:
        start: Startpunkt x,y,z Koordinaten in m
        end: Endpunkt x,y,z Koordinaten in m
        fps: Frames per Second
        kmax: Framezahl am Endpunkt
    """
    BoatPos: List[ndarray] = []
    delta_v = np.subtract(end,start)/(kmax-1)
    #delta_v = (end-start)/kmax
    for k in range(kmax):
        BoatPos.append(start+(k)*delta_v)
    return BoatPos

  
