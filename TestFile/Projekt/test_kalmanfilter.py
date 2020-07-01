import  matplotlib.pyplot as plt
import numpy as np
import math
#from KalmanFilter import KalmanFilter <- was macht das und warum syntax fehler??
import unittest

class TestKalmanFilter(unittest.TestCase):
    def test_can_initialized(self):
        x = 2
        P = 7

        kf = KalmanFilter(x_0=x, P_0=P)
        self.assertAlmostEqual(kf.state, x)
        self.assertAlmostEqual(kf.cov, P)