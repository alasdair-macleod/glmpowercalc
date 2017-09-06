import numpy as np
from glmpowercalc.unirep import Countr

def AS(irr, lim1, alb, sigma, cc, acc, anc, n, ith, prnt_prob, error_chk):
    """
    Computes distribution of a linear combination of non-central chi-
    squared random variables. Taken from Algorithm AS 155, Applied
    Statistics (1980), Vol 29, No 3.

    :param irr: Number of chi-squared terms in the sum
    :param lim1: Maximum number of integration terms
    :param alb: IRRx1 vector of constant multipliers
    :param sigma: Coefficient of normal term -- ? TODO what is this
    :param cc: Point at which the distribution function should be evaluated
    :param acc: Error bound
    :param anc: Vector of non-centrality parameters
    :param n: Vector of degrees of freedom
    :param ith: Vector of ranks of absolute values of ALB ?? how is this used. is it ever not 1?
    :param prnt_prob: =True or = False, True prints QF
    :param error_chk: =True or =False, True prints ICOUNT, TRACE, IFAULT
    :return:
            QF, Probability that the quadratic form is less than CC
            TRACE, 7x1 vector of variables that indicate the performance
                    of the procedure
               TRACE[1] = Absolute value sum
               TRACE[2] = Total number of integration terms
               TRACE[3] = Number of integrations
               TRACE[4] = Integration interval in main integration
               TRACE[5] = Truncation point in initial integration
               TRACE[6] = Standard deviation of convergence factor term
               TRACE[7] = Number of cycles to locate integration parameters
            IFAULT, Output fault indicator
                    =0      No error
                    =1      Requested accuracy could not be obtained
                    =2      Round-off error possibly significant
                    =3      Invalid parameters
                    =4      Unable to location integration parameters
            ICOUNT, Number of times the function was called
    """
    # defines constants
    aln28 = np.log(2) / 8
    pi = 2 * np.arccos(0)
    ndtsrt = True
    fail = False

    # initialize variables
    ifault = 0
    icount = Countr()
    aintl = 0
    ersm = 0
    qf = -1
    trace = np.zeros(7)

    # produce local copies of some variables
    c = cc
    ir = irr
    lim = lim1
    acc1 = acc

    # AMEAN, Scalar representing the expected value of Q
    # SD, Scalar representing the squared deviation of Q- the second moment
    # ALMAX, Maximum of the constants
    # ALMIN, Minimum of the constants
    xlim = lim
    sigsq = sigma ** 2
    sd = sigsq
    almax = 0
    almin = 0
    amean = 0
    j = 1

    #Check validity of input
    valid, msg = isInputValid(n, anc)[0]
    if not valid:
        raise Exception(msg)
        return None
    # Calculate sum of initial sd + variance for each term in linear combination
    sd = sumVariances(initialVariance = sd,
                      linearCombinationConstantCoeffs=alb,
                      degreesOfFreedom=n,
                      nonCentralities=anc)
    # Calculate sum of means for each term in linear combination
    amean = sumMeans(linearCombinationConstantCoeffs=alb,
                     degreesOfFreedom=n,
                     nonCentralities=anc)


def sumMeans(linearCombinationConstantCoeffs,
            degreesOfFreedom,
            nonCentralities):
    sumofmeans = sum([
        linearCombinationConstantCoeffs[i] * (degreesOfFreedom[i] + nonCentralities[i])
        for i in range(len(degreesOfFreedom))
    ])
    return sumofmeans


def sumVariances(initialVariance,
                linearCombinationConstantCoeffs,
                degreesOfFreedom,
                nonCentralities):
    """
    Calculates the sum of the initial variance (user defined? -- where??) plus the variance for each term in the linear
    combination of non-central chi-squared random variables.

    :param initialVariance: possibly user defined?? where does this come from???
    :param linearCombinationConstantCoeffs: the constant multipliers in the linear combination of non-central
    chi-squared random variables.
    :param: degreesOfFreedom: List (vector) of degrees of freedom (n in IML)
    :param nonCentralities: List (vector) of non centrality parameters in linear combination of non-central
    chi-squared random variables.
    :return: Scalar value representing the sum of the initial variance plus the variance for each term in the linear
    combination of non-central chi-squared random variables.
    """
    variance = initialVariance + sum(
                   [
                        linearCombinationConstantCoeffs[i]**2 *
                        (2*degreesOfFreedom[i] + 4*nonCentralities[i]) for i in range(len(degreesOfFreedom))
                   ]
               )
    return variance



def isInputValid(degreesOfFreedom, nonCentralities):
    """
    Checks whether input for our linear combination of Chi Squared terms is valid and returns a boolean value as appropriate.

    :param degreesOfFreedom:  List (vector) of degrees of freedom (n in IML)
    :param nonCentralities:  List (vector) of non-centrality parameters (anc in IML)
    :return: boolean value describing the validity of the input for linear combination of Chi squared terms
    """
    if not all(i >= 0 for i in degreesOfFreedom):
        return False, 'Cannot have negative degrees of freedom.'
    if not all(i >= 0 for i in nonCentralities):
        return False, 'Non centrality parameters cannot be negative'
    return True, 'OK'