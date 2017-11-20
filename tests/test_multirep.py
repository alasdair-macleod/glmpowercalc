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
        result     = hlt(2, 1, 2, 5, eval_HINVE, 0.5, Constants.MULTI_HLT_MCKEON_OS,
                         Constants.CLTYPE_DESIRED_KNOWN, 5, 2, 0.048, 0.052, 0.0001)
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
        result = pbt(2, 1, 2, 5, eval_HINVE, 0.5, Constants.MULTI_PBT_MULLER,
                     Constants.CLTYPE_DESIRED_KNOWN, 5, 2, 0.048, 0.052, 0.0001)
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
        result = wlk(2, 1, 2, 5, eval_HINVE, 0.5, Constants.MULTI_WLK_RAO,
                     Constants.CLTYPE_DESIRED_KNOWN, 5, 2, 0.048, 0.052, 0.0001)
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
        result = special(2, 1, 2, 5, eval_HINVE, 0.5,
                         Constants.CLTYPE_DESIRED_KNOWN, 5, 2, 0.048, 0.052, 0.0001)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)