from munkres import Munkres
from numpy import inf

hungarian = Munkres()

cost  = [[5, 9, 1,5,inf,inf],
          [10, 3, 2,inf,1,inf],
          [8, 7, 4,inf,inf,10]]

indexes = hungarian.compute(cost)
print(cost)
print(indexes)