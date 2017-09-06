import numpy as np
import sys


def AS(irr, lim1, alb, sigma, cc, acc, anc, n, ith, prnt_prob, error_chk):
    """
    Computes distribution of a linear combination of non-central chi-
    squared random variables. Taken from Algorithm AS 155, Applied
    Statistics (1980), Vol 29, No 3.

    :param irr: Number of chi-squared terms in the sum
    :param lim1: Maximum number of integration terms
    :param alb: IRRx1 vector of constant multipliers
    :param sigma: Coefficient of normal term
    :param cc: Point at which the distribution function should be evaluated
    :param acc: Error bound
    :param anc: Vector of non-centrality parameters
    :param n: Vector of degrees of freedom
    :param ith: Vector of ranks of absolute values of ALB
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

    while j <= ir:
        nj = n[j - 1]
        alj = alb[j - 1]
        ancj = anc[j - 1]
        if nj < 0 or ancj < 0:
            ifault = 3
            break

        sd = sd + (alj ** 2) * (2 * nj + 4 * ancj)
        amean = amean + alj * (nj + ancj)
        if almax >= alj:
            if almin > alj:
                almin = alj
        else:
            almax = alj
        j = j + 1

    if sd == 0:
        if c == 0:
            qf = 1
        else:
            qf = 0

    else:
        if almin == 0 and almax == 0 and sigma == 0:
            ifault = 3
        else:
            sd = np.sqrt(sd)
            if almax < -almin:
                almx = -almin
            else:
                almx = almax

            # Define starting values for modules FINDU and CTFF;
            utx = 16 / sd  # In powerlib, it is 16#inv(sd), matrix form
            up = 4.5 / sd
            un = -up

            # Calculate the Truncation point without any convergence factor
            utx = findu(utx,n,alb,anc,0.5 * acc1,lim,icount,sigsq,ir)

            ##############################
            ## DONE UP TO HERE :) ##
            ##############################

            if c != 0 and almx > 0.07 * sd:
                fail, cfe1 = cfe(n,alb,anc,ith,c,lim,icount,ndtsrt,ir)
                tausq = 0.25 * acc1 / cfe1
                if fail:
                    fail = False
                else:
                    if truncn(n, alb, anc, utx, tausq, lim, icount, sigsq, ir) < 0.2 * acc1:
                        sigsq = sigsq + tausq
                        utx = findu(utx,n,alb,anc,0.25 * acc1,lim,icount,sigsq,ir)
                        trace[5] = np.sqrt(tausq)
            trace[4] = utx
            acc1 = 0.5 * acc1

            while True:
                un, fctff = ctff(un,n,alb,anc,acc1,amean,almin,almax,lim,icount,sigsq,ir)
                d1 = fctff - c
                if d1 < 0:
                    qf = 1
                    break
                else:
                    un, fctff = ctff(un,n,alb,anc,acc1,amean,almin,almax,lim,icount,sigsq,ir)
                    d2 = c - fctff
                    if d2 < 0:
                        qf = 0
                        break
                    else:
                        if d1 <= d2:
                            aintv = d2
                        else:
                            aintv = d1
                        aintv = 2 * (2 * np.arccos(0)) / aintv

                        xnt = utx / aintv
                        xntm = 3.0 / np.sqrt(acc1)
                        if xnt <= xntm * 1.5:
                            break
                        else:
                            if xntm > xlim:
                                ifault = 1
                                break
                            else:
                                ntm = round(xntm, 0)
                                aintv1 = utx / xntm
                                x = 2 * (2 * np.arccos(0)) / aintv1
                                if x <= abs(c):
                                    break
                                else:
                                    fail, cx1 = cfe(n,
                                                    alb,
                                                    anc,
                                                    ith,
                                                    c - x,
                                                    lim,
                                                    icount,
                                                    ndtsrt,
                                                    ir)
                                    fail, cx2 = cfe(n,
                                                    alb,
                                                    anc,
                                                    ith,
                                                    c + x,
                                                    lim,
                                                    icount,
                                                    ndtsrt,
                                                    ir)
                                    tausq = cx1 + cx2
                                    tausq = 0.33 * acc1 / (1.1 * tausq)
                                    if fail:
                                        break
                                    else:
                                        acc1 = 0.67 * acc1
                                        aintl, ersm = integr(n,
                                                             alb,
                                                             anc,
                                                             ntm,
                                                             aintv1,
                                                             tausq,
                                                             False,
                                                             c,
                                                             sigsq,
                                                             ir)
                                        xlim = xlim - xntm
                                        sigsq = sigsq + tausq
                                        trace[2] = trace[2] + 1
                                        trace[1] = trace[1] + ntm + 1
                                        utx = findu(utx,
                                                    n,
                                                    alb,
                                                    anc,
                                                    0.25 * acc1,
                                                    lim,
                                                    icount,
                                                    sigsq,
                                                    ir)
                                        acc1 = 0.75 * acc1

                        trace[3] = aintv
                        if xnt > xlim:
                            ifault = 1
                        else:
                            nt = round(xnt, 0)
                            aintl, ersm = integr(n,
                                                 alb,
                                                 anc,
                                                 nt,
                                                 aintv,
                                                 0,
                                                 True,
                                                 c,
                                                 sigsq,
                                                 ir)
                            trace[2] = trace[2] + 1
                            trace[1] = trace[1] + nt + 1
                            qf = 0.5 - aintl
                            trace[0] = ersm
                            up = ersm

                            x = up + (acc / 10.0)
                            j = 1

                            while j <= 8:
                                if j * x == j * up:
                                    ifault = 2
                                j = j + 2

                            trace[6] = icount.count

    if prnt_prob:
        print(qf)
    if error_chk:
        print(trace, ifault, icount.count)

    return qf, trace, ifault, icount.count


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
