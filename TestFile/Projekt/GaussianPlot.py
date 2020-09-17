
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import cv2

cov = np.diag([100, 30])
mean = 20 * np.ones(2)

def gauss(x, mean, cov, normalized=True):
    if len(x.shape) == 2:
        mean = mean[:, None]
    value = np.exp(-.5 * np.sum(np.sum((x[None, :] - mean[None, :]) * np.linalg.inv(cov)[:, :, None], axis=1) * (x - mean), axis=0))
    if normalized:
        value *= 1/2./pi/np.sqrt(np.linalg.det(cov))
    return value


Nx, Ny = 480, 640
z = np.zeros((Nx, Ny))
for i in range(Nx):
        x_vect = np.zeros((2, Ny))
        x_vect[0, :] = i
        x_vect[1, :] = range(Ny)
        # row based evaluation
        weight = .4
        z[i, :] += weight * gauss(x_vect, mean[:2], cov[:2, :2], normalized=True)
        z[i, :] += .6 * gauss(x_vect, np.array([400, 500]), cov, normalized=True)

# normalize
z /= z.max()
# show image
cv2.imshow('gauss', z)
cv2.waitKey(0)
plt.contourf(z)
plt.show()
