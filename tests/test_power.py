from unittest import TestCase
import numpy as np
from glmpowercalc import unirep
from glmpowercalc.constants import Constants
from glmpowercalc.power import CL, IP, Power


class TestPower(TestCase):
    def test_Power(self):
        expected = 0.9
        actual = Power
        self.assertEqual(expected, actual.power(self))
