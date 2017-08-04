from unittest import TestCase
import numpy as np
from glmpowercalc.calculation_state import CalculationState

class TestFwarn(TestCase):

    def test_fwarn(self):
        """ """
        state = CalculationState(0.01)
        state.fwarn(2, 2)
        expected = np.zeros(23, dtype=np.int)
        expected[5] = expected[5] + 1
        actual = state.powerwarn
        self.assertTrue((expected == actual).all())

    def test_directfwarn(self):
        """"""
        state = CalculationState(0.01)
        state.directfwarn(4)
        expected = np.zeros(23, dtype=np.int)
        expected[3] = expected[3] + 1
        actual = state.powerwarn
        self.assertTrue((expected == actual).all())