from unittest import TestCase
import numpy as np

from glmpowercalc.constants import Constants
from glmpowercalc.power import power
from glmpowercalc.input import Scalar, CalcMethod, Option, CL, IP


class TestPower(TestCase):
    def test_Power(self):
        expected = 0.761158
        c_matrix = np.matrix([[1]])
        beta = np.matrix([[1]])
        sigma = np.matrix([[2]])
        essencex = np.matrix([[1]])
        u_matrix = np.matrix(np.identity(np.shape(beta)[1]))
        theta_zero = np.zeros((np.shape(c_matrix)[0], np.shape(u_matrix)[1]))

        actual = power(essencex, beta, c_matrix, u_matrix, sigma, theta_zero, Scalar(rep_n=10), CalcMethod(), Option(), CL(), IP())
        actual_special = actual.special_power
        actual_hlt = actual.hlt_power
        actual_pbt = actual.pbt_power
        actual_wlk = actual.wlk_power
        actual_un = actual.un_power
        actual_hf = actual.hf_power
        actual_cm = actual.cm_power
        actual_gg = actual.gg_power
        actual_box = actual.box_power
        self.assertAlmostEqual(expected, actual_hlt['power'], places=6)
