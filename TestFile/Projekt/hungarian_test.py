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

#print(cost)
#indexes = hungarian.compute(cost)
#print(cost)
#print(indexes)

#A= [[1,2],[3,4]]
#B= [[5],[6]]
#C = concatenate((A,B),axis=1)
#print(len(C[0]))
T = [[4]]
S = [[10]]
U = [[2]]
##print(np.log(4*math.pi**2*np.linalg.det(S)))
##print(np.transpose(S))
#print(np.linalg.inv(S))
#print(np.linalg.det(S))
print(multi_dot([T,U,S]))
