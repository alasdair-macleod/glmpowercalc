from unittest import TestCase
from glmpowercalc.probf import _normal_approximation


class TestProbf(TestCase):
    def test_get_normal_approximation(self):
        """ Example unit test. needs a better name and lots more tests. """
        expected = (4,0)
        result = _normal_approximation(-7)
        print(result)
        assert(result == expected)
