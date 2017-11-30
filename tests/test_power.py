from unittest import TestCase
import numpy as np

from glmpowercalc.constants import Constants
from glmpowercalc.power import power
from glmpowercalc.input import CalcMethod, Option, CL, IP

class TestPower(TestCase):
    def test_Power(self):
        expected = 0.761158
        c_matrix = np.matrix([[1]])
        beta = np.matrix([[1]])
        sigma = np.matrix([[2]])
        essencex = np.matrix([[1]])
        u_matrix = np.matrix(np.identity(np.shape(beta)[1]))
        theta_zero = np.zeros((np.shape(c_matrix)[0], np.shape(u_matrix)[1]))
        rep_n = 10
        beta_scalar = 0.5
        sigma_scalar = 1
        rho_scalar = 1
        alpha = 0.05
        tolerance = 1e-12
        CalcMethod = CalcMethod()
        Option = Option()
        CL = CL()
        IP = IP()
        actual = power(essencex, beta, c_matrix, u_matrix, sigma, theta_zero, rep_n, beta_scalar, sigma_scalar, rho_scalar, alpha,
          tolerance, CalcMethod, Option, CL, IP)
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
