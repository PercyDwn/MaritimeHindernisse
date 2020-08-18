from munkres import Munkres
import numpy as np
from numpy import inf
from numpy import concatenate
import math
from numpy.linalg import multi_dot
import  matplotlib.pyplot as plt
hungarian = Munkres()

plt.figure('test')
k = 5
for i in range(k):
    y= i**2
    plt.plot(i,y,"x",color= 'orange')
    plt.show()



