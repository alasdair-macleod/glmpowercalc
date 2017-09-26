from unittest import TestCase
import numpy as np
from glmpowercalc import unirep


class TestUnirep(TestCase):
    def test_Firstuni1(self):
        """ The value of the object should equal to expected """
        expected = (2,
                    np.array([1, 1]),
                    0.4310345,
                    np.array([-0.074456, 1.0744563]),
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

    def test_Firstuni2(self):
        """ should raise error """
        with self.assertRaises(Exception):
            actual = unirep.firstuni(sigmastar=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                     rank_U=2)

    def test_hfexeps(self):
        """ should return expected value """
        expected = 0.2901679
        actual = unirep.hfexeps(sigmastar=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5,
                                u_method=2)
        self.assertAlmostEqual(actual, expected, places=7)
