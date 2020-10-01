# Plot
import  matplotlib.pyplot as plt

# System
from sys import stdout
from time import sleep

# Typing
from typing import cast

# Mathematics
import numpy as np
from numpy import vstack
from numpy import array
from numpy import ndarray
from scipy.stats import multivariate_normal
from pyrate_common_math_gaussian import Gaussian

def plotGMM(gmm: list, pixel_w: int, pixel_h: int, detail: int = 1) -> plt:

  rows, cols = pixel_h, pixel_w
  gz = np.zeros((rows, cols))
  
  gaussian_count = len(gmm)
  gaussian_counter = 0
  print('analyse gaussians (' + str(gaussian_count) + ' total)')
  for gaussian_obj in gmm:
    gaussians_analysed_percent = cast(int, gaussian_counter / gaussian_count * 100)
    stdout.write("\r%d percent finished" % gaussians_analysed_percent)
    stdout.flush()
    # print("gaussian " + str(gaussian_counter) + " of " + str(gaussian_count))
    distribution = gaussian_obj.distribution()
    for i in range(1,rows, detail):
      for j in range(1, cols, detail):
        gz[i,j] +=  gaussian_obj.distributionValue(distribution, vstack([j, i, 0, 0]))
    gaussian_counter+=1
  stdout.write("\r100 percent finished")
  stdout.flush()
  stdout.write("\n")

  gz /= gz.max()
  plt.contourf(gz)
  plt.gca().invert_yaxis()

  return plt
