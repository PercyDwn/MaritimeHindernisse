from munkres import Munkres
import numpy as np
from numpy import inf
from numpy import concatenate
import math


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
S = np.array([[10]])
print(np.linalg.det(S)) #Fehler? VS.Lukas: numpy has not attribut det => np.det(S) in np.linalg.det(S) 