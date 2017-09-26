from unittest import TestCase
import numpy as np
from glmpowercalc import unirep


class TestUnirep(TestCase):
    def test_Firstuni(self):
        """ The value of the object should equal to expected """
        expected = (2,
                    np.array([1, 1]),
                    0.4310345,
                    np.array([1.0744563, -0.074456]),
                    1,
                    1.16,
                    1)
        actual = unirep.firstuni(sigmastar=np.matrix([[1, 2], [3, 4]]),
                                 rank_U=2)
        self.assertEqual(actual[0], expected[0])
        self.assertTrue((actual[1] == expected[1]).all())
        self.assertAlmostEqual(actual[2], expected[2], places=7)
        self.assertAlmostEqual(actual[3][0], expected[3][0], delta=0.000001)
        self.assertAlmostEqual(actual[3][1], expected[3][1], delta=0.000001)
        self.assertAlmostEqual(actual[4], expected[4], places=7)
        self.assertAlmostEqual(actual[5], expected[5], places=7)
        self.assertAlmostEqual(actual[6], expected[6], places=7)

