import numpy as np
import sys


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
        y = x / (2 + x)
        term = 2 * y ** 3
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
            s1 = s + term / ak
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
        nj = n[j - 1]
        alj = alb[j - 1]
        ancj = anc[j - 1]
        x = u * alj
        y = 1 - x
        const = const + alj * (ancj / y + nj) / y
        sum1 = sum1 + ancj * ((x / y) ** 2) + nj * ((x ** 2) / y + alog1(-x, False))
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
        rb = 2 * almin
    else:
        rb = 2 * almax

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
        errbound, const = errbd(n, alb, anc, u / (1 + u * rb), lim, icount, sigsq, ir)
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
    sum2 = (sigsq + tausq) * u ** 2
    prod1 = 2 * sum2
    u = 2 * uu
    j = 1

    while j <= ir:
        alj = alb[j - 1]
        ancj = anc[j - 1]
        nj = n[j - 1]
        x = (u * alj) ** 2
        sum1 = sum1 + ancj * x / (1 + x)
        if x > 1:
            prod2 = prod2 + nj * np.log(x)
            prod3 = prod3 + nj * alog1(x, True)
            ns = ns + nj
        else:
            prod1 = prod1 + nj * alog1(x, True)
        j = j + 1

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


def findu(utx, n, alb, anc, accx, lim, icount, sigsq, ir):
    """
    This module finds U such that _TRUNCN(U) < ACCX and _TRUNCN(U / 1.2) > ACCX.
    :param utx:
    :param n: Vector of degrees of freedom
    :param alb: IRx1 vector of constant multipliers
    :param anc: Vector of noncentrality parameters
    :param accx:
    :param lim: Maximum number of integration terms
    :param icount: Count of number of times this module is called
    :param sigsq: square of SIGMA, the coefficient of normal term
    :param ir: Number of chi-squared terms in the sum
    :return: utx
    """
    divis = [2.0, 1.4, 1.2, 1.1]
    ut = utx
    u = utx / 4

    if truncn(n, alb, anc, u, 0, lim, icount, sigsq, ir) > accx:
        u = ut
        while truncn(n, alb, anc, u, 0, lim, icount, sigsq, ir) > accx:
            ut = ut * 4
            u = ut
    else:
        ut = u
        u = u / 4
        while truncn(n, alb, anc, u, 0, lim, icount, sigsq, ir) <= accx:
            ut = u
            u = u / 4

    for i in divis:
        u = ut / i
        if truncn(n, alb, anc, u, 0, lim, icount, sigsq, ir) <= accx:
            ut = u
        else:
            break

    utx = ut

    return utx


def integr(n, alb, anc, nterm, aintrv, tausq, main, c, sigsq, ir):
    """

    :param n: Vector of degrees of freedom
    :param alb: IRx1 vector of constant multipliers
    :param anc: Vector of noncentrality parameters
    :param nterm: Number of terms in integration
    :param aintrv:
    :param tausq:
    :param main: True, False
    :param c: Point at which the distribution function should be evaluated
    :param sigsq: square of SIGMA, the coefficient of normal term
    :param ir: Number of chi-squared terms in the sum
    :return:
            aintl:
            ersm:
    """
    pi = 2 * np.arccos(0)
    ainpi = aintrv / pi
    k = nterm
    aintl = 0
    ersm = 0

    while k >= 0:
        u = (k + 0.5) * aintrv
        sum1 = -2 * u * c
        sum2 = abs(sum1)
        sum3 = -0.5 * sigsq * (u ** 2)
        j = ir
        while j > 0:
            nj = n[j - 1]
            x = 2 * alb[j - 1] * u
            y = x ** 2
            sum3 = sum3 - 0.25 * nj * alog1(y, True)
            y = anc[j - 1] * x / (1 + y)
            z = nj * np.arctan(x) + y
            sum1 = sum1 + z
            sum2 = sum2 + abs(z)
            sum3 = sum3 - 0.5 * x * y
            j = j - 1
        x = ainpi * exp1(sum3) / u
        if not main:
            x = x * (1 - exp1(-0.5 * tausq * u ** 2))

        sum1 = np.sin(0.5 * sum1) * x
        sum2 = 0.5 * sum2 * x
        aintl = aintl + sum1
        ersm = ersm + sum2
        k = k - 1

    return aintl, ersm


def cfe(n, alb, anc, ith, x, lim, icount, ndtsrt, ir):
    """
    This module computes the coefficient of TAUSQ in error when
    the convergence factor of Exp(-0.5 * TAUSQ * U**2) is used when DF
    is evaluated at X.

    :param n: Vector of degrees of freedom
    :param alb: IRx1 vector of constant multipliers
    :param anc: Vector of noncentrality parameters
    :param ith: Vector of ranks of absolute values of ALB
    :param x:
    :param lim: Maximum number of integration terms
    :param icount: Count of number of times this module is called
    :param ndtsrt:   =True if _ORDER module has not been run
                     =False if _ORDER module has been run
    :param ir: Number of chi-squared terms in the sum
    :return:
            fail, =True if module produces unreasonable values
            fcfe, Coefficient of TAUSQ
    """
    icount.counting(lim)

    if ndtsrt:
        ith, ndtsrt = order(alb)

    axl = abs(x)
    sxl = x / abs(x)
    sum1 = 0
    j = ir

    while j > 0:
        it = ith[j - 1]
        if alb[it - 1] * sxl > 0:
            alj = abs(alb[it - 1])
            axl1 = axl - alj * (n[it - 1] + anc[it - 1])
            aln28 = np.log(2) / 8
            axl2 = alj / aln28
            if axl1 > axl2:
                axl = axl1
                j = j - 1
            else:
                if axl > axl2:
                    axl = axl2
                sum1 = (axl - axl1) / alj
                k = j - 1
                while k > 0:
                    itk = ith[k - 1]
                    sum1 = sum1 + (n[itk - 1] + anc[itk - 1])
                    k = k - 1
                break
        else:
            j = j - 1

    if sum1 > 100:
        fcfe = 1
        fail = True
    else:
        fcfe = 2 ** (sum1 / 4) / ((2 * np.arccos(0)) * axl ** 2)
        fail = False  # In powerlib, this case has not been defined

    return fail, fcfe
