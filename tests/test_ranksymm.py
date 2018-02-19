from unittest import TestCase
import numpy as np
from glmpowercalc.ranksymm import ranksymm
from glmpowercalc.exceptions.ranksymm_validation_exception import RanksymmValidationException


class TestRanksymm(TestCase):

    def test_emptymatrix(self):
        """It should raise an exception if matrix is Empty"""
        m = np.matrix([])

        with self.assertRaises(RanksymmValidationException):
            res = ranksymm(m, 1)

    def test_nonsquare(self):
        """It should raise an exception if the matrix is not square"""
        m = np.matrix([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(RanksymmValidationException):
            res = ranksymm(m, 1)

    def test_allmissing(self):
        """It should raise an exception if the matrix are all missing"""
        m = np.matrix([[np.NaN, np.NaN], [np.NaN, np.NaN]])

        with self.assertRaises(RanksymmValidationException):
            res = ranksymm(m, 1)

    def test_matrixnotzero(self):
        """It should raise an exception if we have a zero matrix"""
        m = np.matrix([[0, 0], [0, 0]])

        with self.assertRaises(RanksymmValidationException):
            res = ranksymm(m, 1)

    def test_nonsymm(self):
        """It should raise an exception if we have a non-symmetric matrix"""
        m = np.matrix([[1, 2], [3, 4]])

        with self.assertRaises(RanksymmValidationException):
            res = ranksymm(m, 0.0000000000001)

    def test_negativedefine(self):
        """It should raise an exception if the matrix is negative defined"""
        m = np.matrix([[1, 2], [2, 1]])

        with self.assertRaises(RanksymmValidationException):
            res = ranksymm(m, 0.0000000000001)

    def test_rankmatrix(self):
        m = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expected = 3
        actual = ranksymm(m, 0.0000000000001)
        self.assertEqual(expected, actual)
