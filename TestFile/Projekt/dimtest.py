import numpy as np
from numpy import linalg as lin
import math
import filterpy

R = np.array([0.2])
r = lin.inv(R)

print(R)
print('----------')
print(r)
