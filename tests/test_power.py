from unittest import TestCase
import numpy as np
from glmpowercalc import unirep
from glmpowercalc.constants import Constants
from glmpowercalc.power import CL, IP, Power


class TestPower(TestCase):
    def test_CL(self):
        expected = Constants.CLTYPE_NOT_DESIRED
        actual = CL().cl_type
        self.assertEqual(expected, actual)

    def test_Power(self):
        expected = 0.761158
        actual = Power().power()[1][0][0]
        self.assertAlmostEqual(expected, actual, places=6)
