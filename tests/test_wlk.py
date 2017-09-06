from unittest import TestCase
import numpy as np
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.wlk import wlk
import math


def my_round(n, ndigits):
    part = n * 10 ** ndigits
    delta = part - int(part)
    # always round "away from 0"
    if delta >= 0.5 or -0.5 < delta <= 0:
        part = math.ceil(part)
    else:
        part = math.floor(part)
    return part / (10 ** ndigits)


class TestWlk(TestCase):

    def test_wlk1(self):
        """
        This should return the expected value for m_method[2] = 2
        and min(rank_C, rank_U) = 1
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        expected = (0.52768, 0.69772, 0.86495)
        powerwarn = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result = wlk(2, 1, 2, 5, eval_HINVE, 0.5, [4, 2, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (round(result[0], 5),
                  round(result[1], 5),
                  round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_wlk2(self):
        """
        This should return the expected value for m_method[2] = 1
        and min(rank_C, rank_U) = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        # This result is not very accurate, actual power should be 0.6844333
        expected = (0.52505, 0.68444, 0.85220)
        powerwarn = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result = wlk(2, 2, 2, 5, eval_HINVE, 0.5, [1, 1, 1],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (my_round(result[0], 5),
                  my_round(result[1], 5),
                  my_round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_wlk3(self):
        """
        This should return the expected value for m_method[2] = 2
        and min(rank_C, rank_U) = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        expected = (0.53715, 0.75098, 0.92224)
        powerwarn = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result = wlk(2, 2, 2, 5, eval_HINVE, 0.5, [1, 1, 2],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (round(result[0], 5),
                  round(result[1], 5),
                  round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_wlk4(self):
        """
        This should return the expected value for m_method[2] = 3
        and min(rank_C, rank_U) = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        expected = (0.52505, 0.68444, 0.85220) # the expected power should be 0.68443
        powerwarn = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result = wlk(2, 2, 2, 5, eval_HINVE, 0.5, [1, 1, 3],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (my_round(result[0], 5),
                  my_round(result[1], 5),
                  my_round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_wlk5(self):
        """
        This should return the expected value for m_method[2] = 4
        and min(rank_C, rank_U) = 2
        results are round to 5 decimal places because there is difference starting from
        6th decimal place.
        """
        expected = (0.53715, 0.75098, 0.92224)
        powerwarn = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result = wlk(2, 2, 2, 5, eval_HINVE, 0.5, [1, 1, 4],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (my_round(result[0], 5),
                  my_round(result[1], 5),
                  my_round(result[2], 5))
        self.assertEqual(expected, actual)

    def test_wlk6(self):
        """
        if eval_HINVE is missing, then power will be missing.
        """
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([float('nan')])
        actual = wlk(2, 2, 2, 5, eval_HINVE, 0.5, [1, 1, 4],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        self.assertTrue(np.isnan(actual))

    def test_wlk7(self):
        """
        if df2 <= 0, then power will be missing.
        """
        powerwarn = CalculationState(0.0001)
        eval_HINVE=np.array([float(0.6)])
        actual = wlk(2, 2, 2, 1, eval_HINVE, 0.5, [1, 1, 4],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        self.assertTrue(np.isnan(actual))