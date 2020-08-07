from munkres import Munkres
import numpy as np
from numpy import inf
from numpy import concatenate
import math
from numpy.linalg import multi_dot

hungarian = Munkres()

cost  = [[5, 9, 1,5,inf,inf],
          [10, 3, 2,inf,1,inf],
          [8, 7, 4,inf,inf,10]]

print(np.log(np.exp(2)))

