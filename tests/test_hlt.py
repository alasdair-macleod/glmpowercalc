from unittest import TestCase
import numpy as np
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.hlt import hlt

class TestHlt(TestCase):

    def test_hlt1(self):
        """
        This should return the expected value for m_method[0] = 4
        min_rank_C_U = 1
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
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
        min_rank_C_U = 1
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
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
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
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

    def test_hlt4(self):
        """
        This should return the expected value for m_method[0] = 2
        and min_rank_C_U = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """

        expected   = (np.array([0.5049794]), np.array([0.54206]), np.array([0.60127]))
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result     = hlt(2, 2, 2, 5, eval_HINVE, 0.5, [2, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 7),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_hlt5(self):
        """
        This should return the expected value for m_method[0] = 3
        and min_rank_C_U = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """

        expected   = (np.array([0.51479]), np.array([0.61586]), np.array([0.74643]))
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result     = hlt(2, 2, 2, 5, eval_HINVE, 0.5, [3, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_hlt6(self):
        """
        if eval_HINVE is missing, then power will be missing.
        """
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([float('nan')])
        actual     = hlt(2, 2, 2, 5, eval_HINVE, 0.5, [4, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        self.assertTrue(np.isnan(actual))

    def test_hlt7(self):
        """
        if df2 <= 0, then power will be missing.
        """
        powerwarn  = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        actual     = hlt(2, 2, 2, 1, eval_HINVE, 0.5, [4, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        self.assertTrue(np.isnan(actual))