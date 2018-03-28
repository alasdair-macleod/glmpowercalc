from unittest import TestCase
from glmpowercalc.orpol import orpol
import numpy as np

class TestOrpol(TestCase):

    def test_orpol1(self):
        """
        This should return the expected value
        test case is from Emerson 1968 article, and the expected value is from SAS/IML output
        """
        expected = [[0.25, -0.472456, 0.433013, -0.163663],
                    [0.25, -0.094491, -0.144338, 0.272772],
                    [0.25, 0.0944911, -0.144338, -0.272772],
                    [0.25, 0.4724556, 0.433013, 0.163663]]
        actual = orpol([0, 2, 3, 5], 4, [2, 6, 6, 2])
        result = np.round(actual, 6)
        self.assertTrue((expected == result).all)

    def test_orpol2(self):
        """
        Test for maxdegree < n
        This should return the expected value
        """
        expected = [[0.25, -0.472456, 0.433013],
                    [0.25, -0.094491, -0.144338],
                    [0.25, 0.0944911, -0.144338],
                    [0.25, 0.4724556, 0.433013]]
        actual = orpol([0, 2, 3, 5], 3, [2, 6, 6, 2])
        result = np.round(actual, 6)
        self.assertTrue((expected == result).all)

    def test_orpol_none_maxdegree(self):
        """
        Test for maxdegree not specify
        This should return the expected value
        """
        expected = [[0.25, -0.472456, 0.433013, -0.163663],
                    [0.25, -0.094491, -0.144338, 0.272772],
                    [0.25, 0.0944911, -0.144338, -0.272772],
                    [0.25, 0.4724556, 0.433013, 0.163663]]
        actual = orpol(x=[0, 2, 3, 5], weights=[2, 6, 6, 2])
        result = np.round(actual, 6)
        self.assertTrue((expected == result).all)


    def test_uploy_onefactor(self):
        """

        :return:
        """
        expected = None
        actual = orpol([[1,2,3]])
        self.assertTrue()