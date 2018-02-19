from unittest import TestCase

from glmpowercalc.constants import Constants
from glmpowercalc.glmmpcl import glmmpcl


class TestGlmmpcl(TestCase):

    def test_glmmpcl(self):
        """
        This should return the expected value
        """

        expected = (0.9999379, 1, Constants.FMETHOD_NOAPPROXIMATION, Constants.FMETHOD_NOAPPROXIMATION, 105.66408, 315.62306)
        result = glmmpcl(alphatest=0.05,
                         dfh=20,
                         n2=30,
                         dfe2=28,
                         cl_type=Constants.CLTYPE_DESIRED_KNOWN,
                         n_est=20,
                         rank_est=1,
                         alpha_cl=0.048,
                         alpha_cu=0.052,
                         tolerance=0.01,
                         power=0.9,
                         omega=200)
        actual = (round(result[0], 7),
                  result[1],
                  result[2],
                  result[3],
                  round(result[4], 5),
                  round(result[5], 5))
        self.assertEqual(expected, actual)
