from unittest import TestCase
import numpy as np
from glmpowercalc.multirep import hlt, pbt, wlk, special
from glmpowercalc.constants import Constants
from glmpowercalc.input import Scalar, CL

class TestMultirep(TestCase):

    def test_hlt(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        cl = CL(cl_desire=True, sigma_type=True, n_est=5, rank_est=2)
        result = hlt(2, 1, 2, 5, eval_HINVE, Constants.MULTI_HLT_MCKEON_OS, cl, Scalar())
        actual = (np.round(result['lower'], 5),
                  np.round(result['power'], 5),
                  np.round(result['upper'], 4))
        self.assertEqual(expected, actual)

    def test_pbt(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE=np.array([0.6])
        cl = CL(cl_desire=True, sigma_type=True, n_est=5, rank_est=2)
        result = pbt(2, 1, 2, 5, eval_HINVE, Constants.MULTI_PBT_MULLER, cl, Scalar())
        actual = (np.round(result['lower'], 5),
                  np.round(result['power'], 5),
                  np.round(result['upper'], 4))
        self.assertEqual(expected, actual)

    def test_wlk(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        cl = CL(cl_desire=True, sigma_type=True, n_est=5, rank_est=2)
        result = wlk(2, 1, 2, 5, eval_HINVE, Constants.MULTI_WLK_RAO, cl, Scalar())
        actual = (np.round(result['lower'], 5),
                  np.round(result['power'], 5),
                  np.round(result['upper'], 4))
        self.assertEqual(expected, actual)


    def test_special(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        cl = CL(cl_desire=True, sigma_type=True, n_est=5, rank_est=2)
        result = special(2, 1, 2, 5, eval_HINVE, cl, Scalar())
        actual = (np.round(result['lower'], 5),
                  np.round(result['power'], 5),
                  np.round(result['upper'], 4))
        self.assertEqual(expected, actual)