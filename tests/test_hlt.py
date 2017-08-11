from unittest import TestCase
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.hlt import hlt

class TestHlt(TestCase):

    def test_hlt(self):
        """
        This should return the expected value
        """

        expected = (0.527681, 0.6977186, 0.8649466)
        powerwarn = CalculationState(0.0001)
        result = hlt(2, 1, 2, 5, 0.6, 0.5, [4, 2, 2], [0,0,0,0,0],
                     1, 5, 2, 0.048, 0.052, 0.0001, powerwarn)
        actual = (round(result[0], 7),
                round(result[1], 7),
                round(result[2], 7))
        self.assertEqual(expected, actual)