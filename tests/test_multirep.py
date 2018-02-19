from unittest import TestCase
import numpy as np
from glmpowercalc.multirep import hlt, pbt, wlk, special
from glmpowercalc.constants import Constants

class TestMultirep(TestCase):

    def test_hlt(self):
        """
        This should return the expected value
        """

        expected   = (np.array([0.52768]), np.array([0.69772]), np.array([0.86495]))
        eval_HINVE = np.array([0.6])
        CL= CL()
        result     = hlt(2, 1, 2, 5, eval_HINVE, Constants.MULTI_HLT_MCKEON_OS, CL, Scalar)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_pbt(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.52768]), np.array([0.69772]), np.array([0.86495]))
        eval_HINVE=np.array([0.6])
        result = pbt(2, 1, 2, 5, eval_HINVE, Constants.MULTI_PBT_MULLER,
                     0.048)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_wlk(self):
        """
        This should return the expected value
        """

        expected = (0.52768, 0.69772, 0.86495)
        eval_HINVE = np.array([0.6])
        result = wlk(2, 1, 2, 5, eval_HINVE, Constants.MULTI_WLK_RAO,
                     0.0001)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)


    def test_special(self):
        """
        This should return the expected value
        """

        expected = (0.52768, 0.69772, 0.86495)
        eval_HINVE = np.array([0.6])
        result = special(2, 1, 2, 5, eval_HINVE)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)