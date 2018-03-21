from unittest import TestCase
from glmpowercalc.orpol import orpol

class TestOrpol(TestCase):

    def test_orpol1(self):
        """
        This should return the expected value
        """
        expected = []
        result = orpol([0, 2, 3, 5], 4, [2, 6, 6, 2])
        self.assertEqual(expected, result)