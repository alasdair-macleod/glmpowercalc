from unittest import TestCase
import numpy as np
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.pbt import pbt

class TestPbt(TestCase):

    def test_pbt1(self):
        """
        This should return the expected value for m_method[1] = 2
        and min(rank_C, rank_U) = 1
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
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
        and min(rank_C, rank_U) = 1
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
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
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
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

    def test_pbt4(self):
        """
        This should return the expected value for m_method[1] = 2
        and min(rank_C, rank_U) = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        expected = (np.array([0.55363]), np.array([0.83034]), np.array([0.97629]))
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([0.6])
        result = pbt(2, 2, 2, 5, eval_HINVE, 0.5, [1, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_pbt5(self):
        """
        This should return the expected value for m_method[1] = 3
        and min(rank_C, rank_U) = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        expected = (np.array([0.540897]), np.array([0.77032]), np.array([0.93888]))
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([0.6])
        result = pbt(2, 2, 2, 5, eval_HINVE, 0.5, [1, 3, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 6),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_pbt6(self):
        """
        This should return the expected value for m_method[1] = 4
        and min(rank_C, rank_U) = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        expected = (np.array([0.53274]), np.array([0.73306]), np.array([0.91166]))
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([0.6])
        result = pbt(2, 2, 2, 5, eval_HINVE, 0.5, [1, 4, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_pbt7(self):
        """
        if eval_HINVE is missing, then power will be missing.
        """
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([float('nan')])
        actual = pbt(2, 2, 2, 5, eval_HINVE, 0.5, [1, 4, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        self.assertTrue(np.isnan(actual))

    def test_pbt8(self):
        """
        if df2 <= 0, then power will be missing.
        """
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        actual     = pbt(2, 2, 2, 0, eval_HINVE, 0.5, [4, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        self.assertTrue(np.isnan(actual))