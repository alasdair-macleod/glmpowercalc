from unittest import TestCase
import numpy as np
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.pbt import pbt

class TestPbt(TestCase):

    def test_pbt1(self):
        """
        This should return the expected value for m_method[1] = 2
        """
        expected = (np.array([0.52768]), np.array([0.69772]), np.array([0.86495]))
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([0.6])
        result = pbt(2, 1, 2, 5, eval_HINVE, 0.5, [4, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_pbt2(self):
        """
        This should return the expected value for m_method[1] = 1
        """
        expected = (np.array([0.52768]), np.array([0.69772]), np.array([0.86495]))
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([0.6])
        result = pbt(2, 1, 2, 5, eval_HINVE, 0.5, [4, 1, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_pbt3(self):
        """
        This should return the expected value for m_method[1] = 1
        and min(rank_C, rank_U) = 2
        """
        expected = (np.array([0.540897]), np.array([0.77032]), np.array([0.93888]))
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([0.6])
        result = pbt(2, 2, 2, 5, eval_HINVE, 0.5, [1, 1, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 6),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)