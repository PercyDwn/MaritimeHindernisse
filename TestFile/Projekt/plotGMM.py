# Plot
import  matplotlib.pyplot as plt

# System
import os
from sys import stdout

# Typing
from typing import cast

# Mathematics
import numpy as np
from numpy import vstack
from numpy import array
from numpy import ndarray
from scipy.stats import multivariate_normal
from pyrate_common_math_gaussian import Gaussian

import time
import progressbar

def plotGMM(gmm: list, pixel_w: int, pixel_h: int, detail: int = 1, method: str = 'rowwise', figureTitle: str = 'GMM Plot', savePath: str = '') -> plt:

  rows, cols = pixel_h, pixel_w
  gz = np.zeros((rows, cols))
  
  # create new figure
  fig = plt.figure()

  # set count vars and check how many gm there are
  gaussian_count = len(gmm)
  gaussian_counter = 0
  pixel_row_count = gaussian_count * rows
  pixel_row_counter = 0
  print('analyse gaussians (' + str(gaussian_count) + ' total)')
  # calculate gauss values pixelwise
  if method == 'pixelwise':
    for gaussian_obj in gmm:
      # get distribution of current gm
      distribution = gaussian_obj.distribution()
      # iterate over pixels
      for i in range(1,rows, detail):
        for j in range(1, cols, detail):
          # calc gauss value for current pixel and set in gauss array gz
          gz[i,j] +=  gaussian_obj.distributionValue(distribution, vstack([j, i, 0, 0]))
        # increase counter and recalc progress
        pixel_row_counter+=(1*detail)
        pixel_rows_analysed_percent = cast(float,(pixel_row_counter / pixel_row_count) * 100)
        # print progress
        if detail >=5:
          stdout.write("\r%.0f percent finished    " % pixel_rows_analysed_percent)
        elif detail < 5 and detail > 2:
          stdout.write("\r%.1f percent finished    " % pixel_rows_analysed_percent)
        else:
          stdout.write("\r%.2f percent finished    " % pixel_rows_analysed_percent)

    # print finished statement
    stdout.write("\r100 percent finished  \n")
  # calculate gauss values rowwise
  elif method == 'rowwise':
    with progressbar.ProgressBar(max_value=pixel_row_count) as bar:
      for gaussian_obj in gmm:
        # get distribution of current gm
        distribution = gaussian_obj.distribution()
        # iterate over rows
        for i in range(1,rows):
          # set vector with current row and all cols
          x_vect = np.zeros((4, cols))
          x_vect[0, :] = range(cols)
          x_vect[1, :] = i
          # calc gauss value for current row and set in gauss array gz
          gz[i,:] +=  gaussian_obj.distributionValues(distribution, x_vect)
          # increase counter and recalc progress
          pixel_row_counter+=1
          bar.update(pixel_row_counter)

  # normalize
  gz /= gz.max()
  # draw contour
  plt.contourf(gz)
  # set window title
  plt.gcf().canvas.set_window_title(figureTitle)
  # invert y axis
  plt.gca().invert_yaxis()

  # save figure if savePath is set and valid
  if savePath != '' and savePath[0] == '/': 
    path = os.getcwd()
    path = path.replace(os.sep,'/')
    path = path + '/TestFile/Projekt' + savePath 
    if os.path.isdir(path):
      filename = path + '/' + figureTitle + '.png'
      saved = plt.gcf().savefig(fname = filename)
      print('saved figure to ' + filename)
    else:
      print('ERROR can\'t save figure, directory doesn\'t exist!')
      print('invalid path: ' + path)

  return plt




###################################
#ALTE VERSION
####################################
## Plot
#import  matplotlib.pyplot as plt

## System
#from sys import stdout
#from time import sleep

## Typing
#from typing import cast

## Mathematics
#import numpy as np
#from numpy import vstack
#from numpy import array
#from numpy import ndarray
#from scipy.stats import multivariate_normal
#from pyrate_common_math_gaussian import Gaussian

#def plotGMM(gmm: list, pixel_w: int, pixel_h: int, detail: int = 1) -> plt:

#  rows, cols = pixel_h, pixel_w
#  gz = np.zeros((rows, cols))
  
#  gaussian_count = len(gmm)
#  gaussian_counter = 0
#  pixel_row_count = gaussian_count * rows
#  pixel_row_counter = 0
#  print('analyse gaussians (' + str(gaussian_count) + ' total)')
#  for gaussian_obj in gmm:
#    gaussians_analysed_percent = cast(int, gaussian_counter / gaussian_count * 100)
#    distribution = gaussian_obj.distribution()
#    for i in range(1,rows, detail):
#      for j in range(1, cols, detail):
#        gz[i,j] +=  gaussian_obj.distributionValue(distribution, vstack([j, i, 0, 0]))
#      pixel_row_counter+=(1*detail)
#      pixel_rows_analysed_percent = cast(float,(pixel_row_counter / pixel_row_count) * 100)
#      if detail >=5:
#        stdout.write("\r%.0f percent finished    " % pixel_rows_analysed_percent)
#      elif detail < 5 and detail > 2:
#        stdout.write("\r%.1f percent finished    " % pixel_rows_analysed_percent)
#      else:
#        stdout.write("\r%.2f percent finished    " % pixel_rows_analysed_percent)

      
#      stdout.flush()
        
#    gaussian_counter+=1
#  stdout.write("\r100 percent finished  ")
#  stdout.flush()
#  stdout.write("\n")

#  gz /= gz.max()
#  plt.contourf(gz)
#  plt.gca().invert_yaxis()

#  return plt
