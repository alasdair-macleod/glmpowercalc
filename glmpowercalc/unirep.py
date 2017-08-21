import numpy as np
import sys


def qprob(q, Lambda, nu, omega, accuracy):
    """
    This function returns the CDF of a weighted sum of independent
    chi squares via Davies' algorithm.

    :param q: point at which CDF is evaluated: Prob{Q <= q}
                Note: Q is often zero if start from ratio of positively weighted forms
    :param Lambda: Nx1 vector of coefficients
    :param nu: Nx1 vector of degrees of freedom
    :param omega: Nx1 vector of noncentralities
    :param accuracy: maximum error in probability allowed
    :return: prob, Prob{Q <= q}, the CDF. This is set to zero if any problems occur
    """
    pass


class Countr(object):
    """ object to hold number of calls of Countr"""

    def __init__(self):
        self.count = 0

    def counting(self, lim):
        """
        This module counts number of calls to ERRBD(), TRUNCN(), and CFE().
        :param lim: Maximum number of integration terms
        :return:
        """
        if self.count > lim:
            sys.exit("QF: CANNOT LOCATE INTEGRATION PARAMETERS")
        else:
            self.count = self.count + 1


def alog1(x, first):
    """
    For a number X, this function computes either LN(1+x) or LN(1+x)-x

    :param x: Number on which to perform computation
    :param first:
            = TRUE, when this module is called for a matrix for the first time
            = FALSE , otherwise
    :return: alog1
    """

    if abs(x) <= 0.1:
        y = x/(2 + x)
        term = 2*y**3
        ak = 3.0
        if not first:
            s = -x
        else:
            s = 2.0
        s = s * y
        y = y ** 2
        s1 = s + term / ak
        while s1 != s:
            ak = ak + 2
            term = term * y
            s = s1
            s1 = s + term /ak
        alog1 = s
    elif not first:
        alog1 = np.log(1 + x) - x
    else:
        alog1 = np.log(1 + x)
    return alog1


def exp1(x):
    """
    This function computes e^X for X > -706.
    :param x: scalar
    :return:
            0, if X <= -706
            e^X, if X > -706
    """
    if x <= -706:
        return 0
    else:
        return np.exp(x)


def order(alb):
    """
    This module finds the ranks of absolute values of elements of ALB.
    Ties are ranked arbitrarily, e.g., the matrix {2,2} is ranked {1,2}.

    :param alb: IRx1 vector of constant multipliers, np.array
    :return:
            ITH, Vector of ranks of absolute values of ALB
            NDTSRT, = False if this module has been run
    """
    ith = abs(alb).argsort().argsort()
    ndtsrt = False
    return ith, ndtsrt


def errbd(n, alb, anc, uu, lim, icount, sigsq, ir):
    """
    This module finds bound on tail probability using moment-generating function

    :param n: Vector of degrees of freedom
    :param alb: IRx1 vector of constant multipliers
    :param anc: Vector of noncentrality parameters
    :param uu:
    :param lim: Maximum number of integration terms
    :param icount: Count of number of times this module is called
    :param sigsq: square of SIGMA, the coefficient of normal term
    :param ir: Number of chi-squared terms in the sum
    :return:
            ERRBD, Bound on tail probability
            CX,
    """
    icount.counting(lim)

    const = uu * sigsq
    sum1 = uu * const
    u = 2 * uu
    j = ir

    while j > 0:
        nj = n[j-1]
        alj = alb[j-1]
        ancj = anc[j-1]
        x = u * alj
        y = 1 - x
        const = const + alj * (ancj / y + nj) / y
        sum1 = sum1 + ancj * ((x/y)**2) + nj*((x**2)/y + alog1(-x, False))
        j = j - 1

    errbd = exp1(-0.5 * sum1)
    cx = const

    return errbd, cx


def ctff(upn, n, alb, anc, accx, amean, almin, almax, lim, icount, sigsq, ir):
    """
    This module finds CTF so that:
      If UPN > 0:     P(QF > CTFF) < ACCX
      Otherwise:      P(QF < CTFF) < ACCX

    :param unp:
    :param n: Vector of degrees of freedom
    :param alb: IRx1 vector of constant multipliers
    :param anc: Vector of noncentrality parameters
    :param accx: Error bound
    :param amean: Scalar representing the expected value of the QF
    :param almin: Minimum of the constant multipliers
    :param almax: Maximum of the constant multipliers
    :param lim: Maximum number of integration terms
    :param icount: Count of number of times this module is called
    :param sigsq: square of SIGMA, the coefficient of normal term
    :param ir: Number of chi-squared terms in the sum
    :return:
            upn:
            fctff:
    """

    u2 = upn
    u1 = 0
    c1 = amean

    if u2 <= 0:
        rb = 2*almin
    else:
        rb = 2*almax

    u = u2 / (1 + u2 * rb)
    errbound, c2 = errbd(n, alb, anc, u, lim, icount, sigsq, ir)

    if errbound > accx:
        u1 = u2
        c1 = c2
        u2 = 2 * u2
        u = u2 / (1 + u2 * rb)
    else:
        u = (c1 - amean) / (c2 - amean)

    while u < 0.9:
        u = (u1 + u2) / 2
        errbound, const = errbd(n, alb, anc, u/(1 + u*rb), lim, icount,sigsq, ir)
        if errbound > accx:
            u1 = u
            c1 = const
        else:
            u2 = u
            c2 = const
        u = (c1 - amean) / (c2 - amean)

    upn = u2
    fctff = c2

    return upn, fctff


def truncn(n, alb, anc, uu, tausq, lim, icount, sigsq, ir):
    """
    This function bounds integration error due to truncation at U.

    :param n: Vector of degrees of freedom
    :param alb: IRx1 vector of constant multipliers
    :param anc: Vector of noncentrality parameters
    :param uu:
    :param tausq:
    :param lim: Maximum number of integration terms
    :param icount: Count of number of times this module is called
    :param sigsq: square of SIGMA, the coefficient of normal term
    :param ir: Number of chi-squared terms in the sum
    :return: TRUNCN, integration error
    """
    icount.counting(lim)

    u = uu
    sum1 = 0
    prod2 = 0
    prod3 = 0
    ns = 0
    sum2 = (sigsq + tausq) * u**2
    prod1 = 2 * sum2
    u = 2* uu
    j = 1

    while j <= ir:
        alj = alb[j-1]
        ancj = anc[j-1]
        nj = n[j-1]
        x = (u * alj) ** 2
        sum1 = sum1 + ancj * x / (1 + x)
        if x > 1:
            prod2 = prod2 + nj * np.log(x)
            prod3 = prod3 + nj * alog1(x, True)
            ns = ns + nj
        else:
            prod1 = prod1 + nj * alog1(x, True)
        j= j + 1

    sum1 = 0.5 * sum1
    prod2 = prod1 + prod2
    prod3 = prod1 + prod3
    x = (exp1(-sum1 - 0.25 * prod2)) / (2 * np.arccos(0))
    y = (exp1(-sum1 - 0.25 * prod3)) / (2 * np.arccos(0))

    if ns == 0:
        err1 = 1
    else:
        err1 = x * 2 / ns

    if prod3 > 1:
        err2 = 2.5 * y
    else:
        err2 = 1

    if err2 < err1:
        err1 = err2
    else:
        x = 0.5 * sum2

    if x <= y:
        err2 = 1
    else:
        err2 = y / x

    if err1 < err2:
        truncn = err1
    else:
        truncn = err2

    return truncn
