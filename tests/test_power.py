from unittest import TestCase
import numpy as np
from glmpowercalc import unirep
from glmpowercalc.constants import Constants
from glmpowercalc.power import Power


class TestPower(TestCase):
    def TestPower(self):
        expected = 0.9
        actual = Power().power
        self.assertEqual(expected, actual)