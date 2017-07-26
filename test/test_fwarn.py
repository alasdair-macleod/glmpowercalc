from unittest import TestCase
import numpy as np
from glmpowercalc.fwarn import fwarn


class TestFwarn(TestCase):

    def test_fwarn(self):
        """ """
        expected = np.zeros((23, 1))
        expected[5, 0] = expected[5, 0] + 1
        actual = fwarn(2, 2)
        self.assertTrue((expected == actual).all())