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


class TestHlt(TestCase):

    def test_wlk1(self):
        """
        This should return the expected value for m_method[2] = 2
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
        This should return the expected value for m_method[2] = 2
        and min(rank_C, rank_U) = 2
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

    def test_wlk3(self):
        """
        This should return the expected value for m_method[2] = 1
        and min(rank_C, rank_U) = 2
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
