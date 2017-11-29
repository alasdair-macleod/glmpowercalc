from unittest import TestCase
import numpy as np

from glmpowercalc.constants import Constants
from glmpowercalc.power import CL, IP, Power


class TestPower(TestCase):
    def test_CL(self):
        expected = Constants.CLTYPE_NOT_DESIRED
        actual = CL().cl_type
        self.assertEqual(expected, actual)

    def test_Power(self):
        expected = 0.761158
        actual = Power().power()
        actual_special = actual.special_power
        actual_hlt = actual.hlt_power
        actual_pbt = actual.pbt_power
        actual_wlk = actual.wlk_power
        actual_un = actual.un_power
        actual_hf = actual.hf_power
        actual_cm = actual.cm_power
        actual_gg = actual.gg_power
        actual_box = actual.box_power
        self.assertAlmostEqual(expected, actual, places=6)
