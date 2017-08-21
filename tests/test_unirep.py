from unittest import TestCase
import numpy as np
from glmpowercalc import unirep


class TestUnirep(TestCase):

    def test_Countr(self):
        """ The value of the object should equal to expected"""
        icount = unirep.Countr()
        icount.counting(5)
        expected = 1
        actual = icount.count
        self.assertEqual(actual, expected)

    def test_alog1_1(self):
        """ for abs(x) <= 0.1  and first = True """
        expected = 0.0099503
        actual = unirep.alog1(0.01, True)
        result = round(actual, 7)
        self.assertEqual(result, expected)

    def test_alog1_2(self):
        """ for abs(x) <= 0.1  and first = False """
        expected = -0.00005
        actual = unirep.alog1(0.01, False)
        result = round(actual, 5)
        self.assertEqual(result, expected)

    def test_alog1_3(self):
        """ for abs(x) > 0.1  and first = True """
        expected = 0.6931472
        actual = unirep.alog1(1, True)
        result = round(actual, 7)
        self.assertEqual(result, expected)

    def test_alog1_4(self):
        """ for abs(x) > 0.1  and first = False """
        expected = -0.306853
        actual = unirep.alog1(1, False)
        result = round(actual, 6)
        self.assertEqual(result, expected)

    def test_exp1(self):
        """ should return expected value """
        expected = 0
        actual = unirep.exp1(-706)
        self.assertEqual(actual, expected)

    def test_order(self):
        """ should return expected value """
        expected = (np.array([1, 3, 6, 5, 2, 4]) - 1, False)
        actual = unirep.order(np.array([1, 3, 9, 6, 2, 5]))
        self.assertTrue((actual[0] == expected[0]).all())
        self.assertEqual(actual[1], expected[1])

    def test_errbd(self):
        """ should return expected value """
        expected = (0.0050939, 8.1605556)
        actual = unirep.errbd(n=[2, 3, 5],
                              alb=[2, 3, 4],
                              anc=[1, 2, 3],
                              uu=-0.5,
                              lim=10,
                              icount=unirep.Countr(),
                              sigsq=1,
                              ir=3)
        result = (round(actual[0], 7),
                  round(actual[1], 7))
        self.assertEqual(result, expected)

    def test_ctff(self):
        """ should return expected value """
        expected = (-0.25, 11.09186)
        actual = unirep.ctff(upn=-0.5,
                             n=[2, 3, 5],
                             alb=[2, 3, 4],
                             anc=[1, 2, 3],
                             accx=0.05,
                             amean=1,
                             almin=0.6,
                             almax=0.8,
                             lim=10,
                             icount=unirep.Countr(),
                             sigsq=1,
                             ir=3)
        result = (actual[0], round(actual[1], 5))
        self.assertEqual(result, expected)