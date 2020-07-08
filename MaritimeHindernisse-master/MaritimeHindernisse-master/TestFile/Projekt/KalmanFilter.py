import  matplotlib.pyplot as plt
import numpy as np
import math

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z

class KalmanFilter(object):
  """description of class"""

def __init__(self, dim_x, dim_z, dim_u=0):
    #if dim_x < 1:
    #    raise ValueError('dim_x must be 1 or greater')
    #if dim_z < 1:
    #    raise ValueError('dim_z must be 1 or greater')
    #if dim_u < 0:
    #    raise ValueError('dim_u must be 0 or greater')

    self.dim_x = dim_x
    self.dim_z = dim_z
    self.dim_u = dim_u

    self.x = zeros((dim_x, 1))        # state
    self.P = eye(dim_x)               # uncertainty covariance
    self.Q = eye(dim_x)               # process uncertainty
    self.B = None                     # control transition matrix
    self.F = eye(dim_x)               # state transition matrix
    self.H = zeros((dim_z, dim_x))    # Measurement function
    self.R = eye(dim_z)               # state uncertainty
    self._alpha_sq = 1.               # fading memory control
    self.M = np.zeros((dim_x, dim_z)) # process-measurement cross correlation
    self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
    self.K = np.zeros((dim_x, dim_z)) # kalman gain
    self.y = zeros((dim_z, 1))
    self.S = np.zeros((dim_z, dim_z)) # system uncertainty
    self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
    self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
    self.x_prior = self.x.copy()
    self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
    self.x_post = self.x.copy()
    self.P_post = self.P.copy()

        # Only computed only if requested via property
    self._log_likelihood = log(sys.float_info.min)
    self._likelihood = sys.float_info.min
    self._mahalanobis = None

    self.inv = np.linalg.inv

    #Systembeschreibung:
    # x_kp = F*x_k + B*u_k + n_x,k      Q=Cov(n_x)
    # y_k = H*x_k + n_y,k               R=Cov(n_y)
    
def predict(self, u=None, B=None, F=None, Q=None):
    """
    Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
    """
    if B is None:
        B = self.B
    if F is None:
        F = self.F
    if Q is None:
        Q = self.Q
    elif isscalar(Q):
        Q = eye(self.dim_x) * Q


        # x = Fx + Bu
    if B is not None and u is not None:
        self.x = dot(F, self.x) + dot(B, u)
    else:
         self.x = dot(F, self.x)

        # P = FPF' + Q
    self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
    self.x_prior = self.x.copy()
    self.P_prior = self.P.copy()

def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

def get_prediction(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations and returns it without modifying the object.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        Returns
        -------
        (x, P) : tuple
            State vector and covariance array of the prediction.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            x = dot(F, self.x) + dot(B, u)
        else:
            x = dot(F, self.x)

        # P = FPF' + Q
        P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        return x, P

def get_update(self, z=None):
        """
        Computes the new estimate based on measurement `z` and returns it
        without altering the state of the filter.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        Returns
        -------
        (x, P) : tuple
            State vector and covariance array of the update.
       """

        if z is None:
            return self.x, self.P
        z = reshape_z(z, self.dim_z, self.x.ndim)

        R = self.R
        H = self.H
        P = self.P
        x = self.x

        # error (residual) between measurement and prediction
        y = z - dot(H, x)

        # common subexpression for speed
        PHT = dot(P, H.T)

        # project system uncertainty into measurement space
        S = dot(H, PHT) + R

        # map system uncertainty into kalman gain
        K = dot(PHT, self.inv(S))

        # predict new x with residual scaled by the kalman gain
        x = x + dot(K, y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

        return x, P

#@property
#def log_likelihood(self):
#    """
#    log-likelihood of the last measurement.
#    """
#    if self._log_likelihood is None:
#        self._log_likelihood = logpdf(x=self.y, cov=self.S)
#    return self._log_likelihood

#@property
#def likelihood(self):
#    """
#    Computed from the log-likelihood. The log-likelihood can be very
#    small,  meaning a large negative value such as -28000. Taking the
#    exp() of that results in 0.0, which can break typical algorithms
#    which multiply by this value, so by default we always return a
#    number >= sys.float_info.min.
#    """
#    if self._likelihood is None:
#        self._likelihood = exp(self.log_likelihood)
#        if self._likelihood == 0:
#            self._likelihood = sys.float_info.min
#    return self._likelihood









    #def prediction(xhat_k, Phat_k, A, B, u_k, Q):                       #wie kann man fkt mit variablen Eignagnsmparam definieren? -> wenn kein B bzw u_k vorhanden ??
    #    xhat_kp=np.matmul(A, xhat) + np.matmul(B, u_k)                  #Prädiziertes x(k+1)
    #    Phat_kp=np.matmul(A, np.maltmul(Phat_k, np.transpose(A))) + Q   #Prädiziertes P(k+1)
    #    return xhat_kp, Phat_kp   

    #def update(xhat_k, Phat_k, y_k ,C , R):
    #    K_k = (P_hatk.dot(np.transpose(C))).dot(np.linalg.inv(C.dot(Phat_k).dot(C.transpose())))
    #    xtil = xhat_k+K_k.dot(y_k-C.dot(x_hatk))    #Korriegiertes x(k)
    #    Ptil = Phat_k-K_k.dot(C).dot(Phat_k)        #Korrigertes P(k)
    #    return xtil, Ptil
        
  

    #@property 
    #def state(self):
    #    return self.x

    #@property 
    #def cov(self):
    #    return self.P
