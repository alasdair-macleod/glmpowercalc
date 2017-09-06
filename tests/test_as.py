from unittest import TestCase
from glmpowercalc import asfn, unirep
import numpy as np


class TestAs(TestCase):
    #linear combination
    # sigma_Chi_squared = alb_0 ChiSq_0 + alb_1 ChiSq_1 + .....

    def test_isInputValid_valid(self):
        """It should return true if all degrees of freedom and non centralities are greater than or equal to zero"""
        degreesOfFreedom = [0, 1, 2, 3, 4, 5, 6, 7]
        nonCentralities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9]

        expected = True
        actual, msg = asfn.isInputValid(degreesOfFreedom, nonCentralities)
        self.assertEqual(expected, actual)

    def test_isInputValid_negativeDegFrdm(self):
        """It should return true if all degrees of freedom and non centralities are greater than or equal to zero"""
        degreesOfFreedom = [0, -1, 2, 3, 4, 5, 6, 7]
        nonCentralities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9]

        expected = False
        actual, msg = asfn.isInputValid(degreesOfFreedom, nonCentralities)
        self.assertEqual(expected, actual)

    def test_isInputValid_negativeNonCentrality(self):
        """It should return true if all degrees of freedom and non centralities are greater than or equal to zero"""
        degreesOfFreedom = [0, -1, 2, 3, 4, 5, 6, 7]
        nonCentralities = [0, 1, -2, 3, 4, 5, 6, 7, 8, 8, 9]

        expected = False
        actual, msg = asfn.isInputValid(degreesOfFreedom, nonCentralities)
        self.assertEqual(expected, actual)

    def test_sumVariances(self):
        """It should correctly calculate the sum of variances plus initial variance for all terms in the linear
            combination of non-central chi-squared random variables"""
        initialVariance = 2
        linearCombinationConstantCoeffs = [1, 2, 3]
        degreesOfFreedom = [4, 5, 6]
        nonCentralities = [7, 8, 9]

        expected = 638
        actual = asfn.sumVariances(initialVariance, linearCombinationConstantCoeffs, degreesOfFreedom, nonCentralities)
        self.assertEqual(expected, actual)

    def test_sumMeans(self):
        """It should correctly calculate the sum of means for all terms in the linear
            combination of non-central chi-squared random variables"""
        linearCombinationConstantCoeffs = [1, 2, 3]
        degreesOfFreedom = [4, 5, 6]
        nonCentralities = [7, 8, 9]

        expected = 82
        actual = asfn.sumMeans(linearCombinationConstantCoeffs, degreesOfFreedom, nonCentralities)
        self.assertEqual(expected, actual)

