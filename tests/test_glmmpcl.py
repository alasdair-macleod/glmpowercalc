from unittest import TestCase
from glmpowercalc.calculation_state import CalculationState
from glmpowercalc.glmmpcl import glmmpcl


class TestGlmmpcl(TestCase):

    def test_glmmpcl(self):
        """
        This should return the expected value
        """

        expected = (0.9999379, 1, 1, 1, 105.66408, 315.62306)
        powerwarn = CalculationState(0.01)
        result = glmmpcl(f_a=10,
                           alphatest=0.05,
                           dfh=20,
                           n2=30,
                           dfe2=28,
                           cltype=1,
                           n_est=20,
                           rank_est=1,
                           alpha_cl=0.048,
                           alpha_cu=0.052,
                           tolerance=0.01,
                           powerwarn=powerwarn)
        actual = (round(result[0], 7),
                  result[1],
                  result[2],
                  result[3],
                  round(result[4], 5),
                  round(result[5], 5))
        self.assertEqual(expected, actual)
