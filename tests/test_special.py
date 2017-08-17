from unittest import TestCase
import numpy as np
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.special import special

class TestPbt(TestCase):

    def test_pbt(self):
        """
        This should return the expected value
        """

        expected = (0.52768, 0.69772, 0.86495)
        powerwarn = CalculationState(0.0001)
        eval_HINVE = np.array([0.6])
        result = special(2, 1, 2, 5, eval_HINVE, 0.5, [4, 2, 2], [0,0,0,0,0],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (np.round(result[0], 5),
                  np.round(result[1], 5),
                  np.round(result[2], 5))
        self.assertEqual(expected, actual)