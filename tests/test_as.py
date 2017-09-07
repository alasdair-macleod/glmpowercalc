from unittest import TestCase
from glmpowercalc import asfn

class TestAs(TestCase):

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
        degreesOfFreedom = [0, 1, 2, 3, 4, 5, 6, 7]
        nonCentralities = [0, 1, -2, 3, 4, 5, 6, 7, 8, 8, 9]

        expected = False
        actual, msg = asfn.isInputValid(degreesOfFreedom, nonCentralities)
        self.assertEqual(expected, actual)

    def test_invalidInputException_negativeNonCentrality(self):
        """It should raise an exception is input is invalid"""
        linearCombinationSize = 3
        numericIntegratinLimit = 15000
        linearCombinationConstantCoeffs = [0, 0, 0]
        sigma = 0
        cc = 1
        errorBound = 0
        nonCentralities = [-1, 1, 1]
        degreesOfFreedom = [1, 1, 1]
        constCoeffAbsValRanks = [0, 0, 0]

        with self.assertRaises(Exception):
            actual = asfn.AS(irr=linearCombinationSize,
                             lim1=numericIntegratinLimit,
                             alb=linearCombinationConstantCoeffs,
                             sigma=sigma,
                             cc=cc,
                             acc=errorBound,
                             anc=nonCentralities,
                             n=degreesOfFreedom,
                             ith=constCoeffAbsValRanks,
                             prnt_prob=False,
                             error_chk=False)

    def test_AS_validInput(self):
        """It should not raise an exception is input is valid"""
        linearCombinationSize = 3
        numericIntegratinLimit = 15000
        linearCombinationConstantCoeffs = [1, 2, 3]
        sigma = 0
        cc = 1
        errorBound = 0
        nonCentralities = [1, 1, 1]
        degreesOfFreedom = [1, 1, 1]
        constCoeffAbsValRanks = [0, 0, 0]

        asfn.AS(irr=linearCombinationSize,
                 lim1=numericIntegratinLimit,
                 alb=linearCombinationConstantCoeffs,
                 sigma=sigma,
                 cc=cc,
                 acc=errorBound,
                 anc=nonCentralities,
                 n=degreesOfFreedom,
                 ith=constCoeffAbsValRanks,
                 prnt_prob=False,
                 error_chk=False)


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

    def test_alMinMax_greaterThanZero(self):
        """It should give us the largest and smallest values in the vector of coefficients
        or the input values, if they are greater/smaller accordingly."""
        linearCombinationConstantCoeffs = [1, 2, 3]
        minCoeff = 0
        maxCoeff = 0

        expectedMin = 0
        expectedMax = 3
        actualMin, actualMax = asfn.getMinMaxConstCoefficients(linearCombinationConstantCoeffs, minCoeff, maxCoeff)

        self.assertEqual(expectedMax, actualMax)
        self.assertEqual(expectedMin, actualMin)

    def test_alMinMax_oneNeg(self):
        """It should give us the largest and smallest values in the vector of coefficients
        or the input values, if they are greater/smaller accordingly."""
        linearCombinationConstantCoeffs = [-1, 2, 3]
        minCoeff = 0
        maxCoeff = 0

        expectedMin = -1
        expectedMax = 3
        actualMin, actualMax = asfn.getMinMaxConstCoefficients(linearCombinationConstantCoeffs, minCoeff, maxCoeff)

        self.assertEqual(expectedMax, actualMax)
        self.assertEqual(expectedMin, actualMin)

    def test_alMinMax_allNeg(self):
        """It should give us the largest and smallest values in the vector of coefficients
        or the input values, if they are greater/smaller accordingly."""
        linearCombinationConstantCoeffs = [-1, -2, -3]
        minCoeff = 0
        maxCoeff = 0

        expectedMin = -3
        expectedMax = 0
        actualMin, actualMax = asfn.getMinMaxConstCoefficients(linearCombinationConstantCoeffs, minCoeff, maxCoeff)

        self.assertEqual(expectedMax, actualMax)
        self.assertEqual(expectedMin, actualMin)

    def test_alMinMax_allZero(self):
        """It should give us the largest and smallest values in the vector of coefficients
        or the input values, if they are greater/smaller accordingly."""
        linearCombinationConstantCoeffs = [0, 0, 0]
        minCoeff = 0
        maxCoeff = 0

        expectedMin = 0
        expectedMax = 0
        actualMin, actualMax = asfn.getMinMaxConstCoefficients(linearCombinationConstantCoeffs, minCoeff, maxCoeff)

        self.assertEqual(expectedMax, actualMax)
        self.assertEqual(expectedMin, actualMin)

    def test_alMinMax_spread(self):
        """It should give us the largest and smallest values in the vector of coefficients
        or the input values, if they are greater/smaller accordingly."""
        linearCombinationConstantCoeffs = [-1, 0, 1]
        minCoeff = 0
        maxCoeff = 0

        expectedMin = -1
        expectedMax = 1
        actualMin, actualMax = asfn.getMinMaxConstCoefficients(linearCombinationConstantCoeffs, minCoeff, maxCoeff)

        self.assertEqual(expectedMax, actualMax)
        self.assertEqual(expectedMin, actualMin)

    def test_asfn_zerosSumVariances_ccZero(self):
        """It should give a probabilty of one if the sum of variances is zero and we are evaluating at zero."""
        linearCombinationSize = 3
        numericIntegratinLimit = 15000
        linearCombinationConstantCoeffs = [0,0,0]
        sigma = 0
        cc = 0
        errorBound = 0
        nonCentralities = [1,1,1]
        degreesOfFreedom = [1,1,1]
        constCoeffAbsValRanks = [0,0,0]

        expected = 1
        actual = asfn.AS(irr=linearCombinationSize,
                         lim1=numericIntegratinLimit,
                         alb=linearCombinationConstantCoeffs,
                         sigma=sigma,
                         cc=cc,
                         acc=errorBound,
                         anc=nonCentralities,
                         n=degreesOfFreedom,
                         ith=constCoeffAbsValRanks,
                         prnt_prob=False,
                         error_chk=False)

        self.assertEqual(expected, actual)

    def test_asfn_zerosSumVariances_ccNonZero(self):
        """It should give a probabilty of zero if the sum of variances is zero and we are evaluating at zero."""
        linearCombinationSize = 3
        numericIntegratinLimit = 15000
        linearCombinationConstantCoeffs = [0,0,0]
        sigma = 0
        cc = 1
        errorBound = 0
        nonCentralities = [1,1,1]
        degreesOfFreedom = [1,1,1]
        constCoeffAbsValRanks = [0,0,0]

        expected = 0
        actual = asfn.AS(irr=linearCombinationSize,
                         lim1=numericIntegratinLimit,
                         alb=linearCombinationConstantCoeffs,
                         sigma=sigma,
                         cc=cc,
                         acc=errorBound,
                         anc=nonCentralities,
                         n=degreesOfFreedom,
                         ith=constCoeffAbsValRanks,
                         prnt_prob=False,
                         error_chk=False)

        self.assertEqual(expected, actual)




