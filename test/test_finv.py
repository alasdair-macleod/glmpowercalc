from unittest import TestCase
import numpy as np
from scipy.stats import f
from glmpowercalc.finv import finv

class TestFinv(TestCase):

    def test_largedf(self):
        actual = finv(0.05, 10**8, 1)
        self.assertTrue(np.isnan(actual))

    def test_negativedf(self):
        actual = finv(0.05, -1, 1)
        self.assertTrue(np.isnan(actual))

    def test_finv(self):
        expected = 0.7185356
        actual = finv(0.05, 100, 100)
        result = round(actual, 7)
        self.assertEquals(expected, result)