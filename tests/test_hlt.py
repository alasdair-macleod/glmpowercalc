from unittest import TestCase
import numpy as np
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.hlt import hlt

class TestHlt(TestCase):

    def test_hlt1(self):
        """
        This should return the expected value for m_method[0] = 4
        """

        expected   = (np.array([0.52768]), np.array([0.69772]), np.array([0.86495]))
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result     = hlt(2, 1, 2, 5, eval_HINVE, 0.5, [4, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_hlt2(self):
        """
        This should return the expected value for m_method[0] = 1
        """

        expected   = (np.array([0.52768]), np.array([0.69772]), np.array([0.86495]))
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result     = hlt(2, 1, 2, 5, eval_HINVE, 0.5, [1, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_hlt3(self):
        """
        This should return the expected value for m_method[0] = 1
        and min_rank_C_U = 2
        """

        expected   = (np.array([0.5049794]), np.array([0.54206]), np.array([0.60127]))
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result     = hlt(2, 2, 2, 5, eval_HINVE, 0.5, [1, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 7),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)