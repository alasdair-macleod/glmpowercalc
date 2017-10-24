import numpy as np
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf
from scipy.stats import chi2


def firstuni(sigmastar, rank_U):
    """
    This module produces matrices required for Geisser-Greenhouse,
    Huynh-Feldt or uncorrected repeated measures power calculations. It
    is the first step. Program uses approximations of expected values of
    epsilon estimates due to Muller (1985), based on theorem of Fujikoshi
    (1978). Program requires that U be orthonormal and orthogonal to a
    columns of 1's.

    :param sigmastar: U` * (SIGMA # SIGSCALTEMP) * U
    :param rank_U: rank of U matrix

    :return:
        d, number of distinct eigenvalues
        mtp, multiplicities of eigenvalues
        eps, epsilon calculated from U`*SIGMA*U
        deigval, first eigenvalue
        slam1, sum of eigenvalues squared
        slam2, sum of squared eigenvalues
        slam3, sum of eigenvalues
    """

    if rank_U != np.shape(sigmastar)[0]:
        raise Exception("rank of U should equal to nrows of sigmastar")

    # Get eigenvalues of covariance matrix associated with E. This is NOT
    # the USUAL sigma. This cov matrix is that of (Y-YHAT)*U, not of (Y-YHAT).
    # The covariance matrix is normalized to minimize numerical problems
    esig = sigmastar / np.trace(sigmastar)
    seigval = np.linalg.eigvals(esig)
    slam1 = np.sum(seigval) ** 2
    slam2 = np.sum(np.square(seigval))
    slam3 = np.sum(seigval)
    eps = slam1 / (rank_U * slam2)

    deigval_array, mtp_array = np.unique(seigval, return_counts=True)
    d = len(deigval_array)

    deigval = np.matrix(deigval_array).T
    mtp = np.matrix(mtp_array).T

    return d, mtp, eps, deigval, slam1, slam2, slam3


def hfexeps(sigmastar, rank_U, total_N, rank_X, u_method):
    """

    Univariate, HF STEP 2:
    This function computes the approximate expected value of
    the Huynh-Feldt estimate.

      FK  = 1st deriv of FNCT of eigenvalues
      FKK = 2nd deriv of FNCT of eigenvalues
      For HF, FNCT is epsilon tilde

    :param sigmastar:
    :param rank_U:
    :param total_N:
    :param rank_X:
    :param u_method:
    :return:
    """
    d, mtp, eps, deigval, slam1, slam2, slam3 = firstuni(sigmastar=sigmastar,
                                                         rank_U=rank_U)

    # Compute approximate expected value of Huynh-Feldt estimate
    h1 = total_N * slam1 - 2 * slam2
    h2 = (total_N - rank_X) * slam2 - slam1
    derh1 = np.full((d, 1), 2 * total_N * slam3) - 4 * deigval
    derh2 = 2 * (total_N - rank_X) * deigval - np.full((d, 1), 2 * np.sqrt(slam1))
    fk = (derh1 - h1 * derh2 / h2) / (rank_U * h2)
    der2h1 = np.full((d, 1), 2 * total_N - 4)
    der2h2 = np.full((d, 1), 2 * (total_N - rank_X) - 2)
    fkk = (
              np.multiply(-derh1, derh2) / h2
              + der2h1
              - np.multiply(derh1, derh2) / h2
              + 2 * h1 * np.power(derh2, 2) / h2 ** 2
              - h1 * der2h2 / h2
          ) / (h2 * rank_U)
    t1 = np.multiply(np.multiply(fkk, np.power(deigval, 2)), mtp)
    sum1 = np.sum(t1)

    if d == 1:
        sum2 = 0
    else:
        t2 = np.multiply(np.multiply(fk, deigval), mtp)
        t3 = np.multiply(deigval, mtp)
        tm1 = t2 * t3.T
        t4 = deigval * np.full((1, d), 1)
        tm2 = t4 - t4.T
        tm2inv = 1 / (tm2 + np.identity(d)) - np.identity(d)
        tm3 = np.multiply(tm1, tm2inv)
        sum2 = np.sum(tm3)

    # Define HF Approx E(.) for Method 0
    e0epshf = h1 / (rank_U * h2) + (sum1 + sum2) / (total_N - rank_X)

    # Computation of EXP(T1) and EXP(T2)
    esig = sigmastar / np.trace(sigmastar)
    seval = np.matrix(np.linalg.eigvals(esig)).T

    nu = total_N - rank_X
    expt1 = 2 * nu * slam2 + nu ** 2 * slam1
    expt2 = nu * (nu + 1) * slam2 + nu * np.sum(seval * seval.T)

    # For use with Method 1
    num01 = (1 / rank_U) * ((nu + 1) * expt1 - 2 * expt2)
    den01 = nu * expt2 - expt1

    # Define HF Approx E(.) for Method 1
    e1epshf = num01 / den01

    # u_method
    # =1 --> Muller and Barton (1989) approximation
    # =2 --> Method 1, Muller, Edwards, and Taylor (2004)
    if u_method == 1:
        exeps = e0epshf
    elif u_method == 2:
        exeps = e1epshf

    return exeps


def cmexeps(sigmastar, rank_U, total_N, rank_X, u_method):
    """
    Univariate, HF STEP 2 with Chi-Muller:
    This function computes the approximate expected value of
    the Huynh-Feldt estimate with the Chi-Muller results


    :param sigmastar:
    :param rank_U:
    :param total_N:
    :param rank_X:
    :param u_method:
    :return:
    """

    exeps = hfexeps(sigmastar=sigmastar,
                    rank_U=rank_U,
                    total_N=total_N,
                    rank_X=rank_X,
                    u_method=u_method)

    if total_N - rank_X == 1:
        uefactor = 1
    else:
        nu_e = total_N - rank_X
        nu_a = (nu_e - 1) + nu_e * (nu_e - 1) / 2
        uefactor = (nu_a - 2) * (nu_a - 4) / (nu_a ** 2)

    exeps = uefactor * exeps

    return exeps


def ggexeps(sigmastar, rank_U, total_N, rank_X, u_method):
    """
    Univariate, GG STEP 2:
    This function computes the approximate expected value of the
    Geisser-Greenhouse estimate.

    :param sigmastar:
    :param rank_U:
    :param total_N:
    :param rank_X:
    :param u_method:
    :return:
    """
    d, mtp, eps, deigval, slam1, slam2, slam3 = firstuni(sigmastar=sigmastar,
                                                         rank_U=rank_U)

    fk = np.full((d, 1), 1) * 2 * slam3 / (slam2 * rank_U) - 2 * deigval * slam1 / (rank_U * slam2 ** 2)
    c0 = 1 - slam1 / slam2
    c1 = -4 * slam3 / slam2
    c2 = 4 * slam1 / slam2 ** 2
    fkk = 2 * (c0 * np.full((d, 1), 1) + c1 * deigval + c2 * np.power(deigval, 2)) / (rank_U * slam2)
    t1 = np.multiply(np.multiply(fkk, np.power(deigval, 2)), mtp)
    sum1 = np.sum(t1)

    if d == 1:
        sum2 = 0
    else:
        t2 = np.multiply(np.multiply(fk, deigval), mtp)
        t3 = np.multiply(deigval, mtp)
        tm1 = t2 * t3.T
        t4 = deigval * np.full((1, d), 1)
        tm2 = t4 - t4.T
        tm2inv = 1 / (tm2 + np.identity(d)) - np.identity(d)
        tm3 = np.multiply(tm1, tm2inv)
        sum2 = np.sum(tm3)

    # Define GG Approx E(.) for Method 0
    e0epsgg = eps + (sum1 + sum2) / (total_N - rank_X)

    # Computation of EXP(T1) and EXP(T2)
    esig = sigmastar / np.trace(sigmastar)
    seval = np.matrix(np.linalg.eigvals(esig)).T

    nu = total_N - rank_X
    expt1 = 2 * nu * slam2 + nu ** 2 * slam1
    expt2 = nu * (nu + 1) * slam2 + nu * np.sum(seval * seval.T)

    # Define GG Approx E(.) for Method 1
    e1epsgg = (1 / rank_U) * (expt1 / expt2)

    # u_method
    # =1 --> Muller and Barton (1989) approximation
    # =2 --> Method 1, Muller, Edwards, and Taylor (2004)
    if u_method == 1:
        exeps = e0epsgg
    elif u_method == 2:
        exeps = e1epsgg

    return exeps


def lastuni(ucdf, powercalc, rank_C, rank_U, total_N, rank_X,
            error_sum_square, hypo_sum_square, sig_type, ip_plan, rank_ip,
            n_est, rank_est, n_ip, sigmastareval, sigmastarevec,
            cltype, alpha_cl, alpha_cu, tolerance, round,
            exeps, eps, alpha_scalar, powerwarn):
    """
    Univariate STEP 3
    This module performs the final step for univariate repeated measures power calculations.

    :param ucdf:
        UCDF, a 1x1, 5x1 or 1x5 vector of choices of calculation
        for UNIREP test statistic CDF.

        If a 5x1 or 1x5 matrix is specified, the elements correspond
        to choices for calculation of the CDF of:

        UCDF[1] --> Uncorrected test
        UCDF[2] --> Huynh-Feldt test
        UCDF[3] --> Huynh-Feldt test with Chi-Muller
        UCDF[4] --> Geisser-Greenhouse test
        UCDF[5] --> Box conservative test
        Where:

        UCDF[J] =1 --> Muller and Barton (1989) approximation
                =2 --> Muller, Edwards and Taylor (2004) approximation
                =3 --> Muller, Edwards and Taylor (2004) exact, via
                  Davies' algorithm. This algorithm may fail -- if it
                  does, this choice will yield missing values.
                =4 --> Muller, Edwards and Taylor (2004) exact as in
                  choice 3, except if the Davies' algorithm fails,
                  this replaces the missing value with the value
                  resulting from choice 2

        Specifying a 1x1 matrix with choice 1, 2, 3, or 4 as described
        above gives the same method for all four tests.

        If UCDF is not specified, the default is {2 2 2 2 2}`.

    :param powercalc: indicates selected power calculation, 5=>UN 6=>HF 7=>CM 8=>GG 9=>Box
    :param rank_C: rank of C
    :param rank_U: rank of U
    :param total_N: total number of observations
    :param rank_X: rank of X
    :param error_sum_square: error sum of squares
    :param hypo_sum_square: hypothesis sum of squares
    :param sig_type: (scalar) choice of whether SIGMA is known or estimated
    :param ip_plan: (scalar) flag to compute power in planning an internal pilot design
    :param rank_ip: (scalar) rank of the design matrix used in the future study (required if IP_PLAN=1)
    :param n_ip: (scalar) # of observations planned for the internal pilot of the future study (required if IP_PLAN=1)
    :param rank_est: (scalar) design matrix rank in analysis which yielded BETA and SIGMA estimates
                        (required if CLTYPE>=1)
    :param n_est: (scalar) # of observations in analysis which yielded BETA and SIGMA estimates
                    (required if CLTYPE>=1 or SIGTYPE=1)
    :param sigmastareval: eigenvalues  of SIGMASTAR=U`*SIGMA*U
    :param sigmastarevec: eigenvectors of SIGMASTAR=U`*SIGMA*U
    :param cltype: (scalar) choice of whether confidence limits produced
    :param alpha_cl: (scalar) lower tail probability for power C.L.
    :param alpha_cu: (scalar) upper tail probability for power C.L.
    :param tolerance: (scalar) value not tolerated, numeric zero, used for checking singularity.
    :param round: (scalar) # of places to round power calculations
    :param exeps: expected value epsilon estimator
    :param eps: epsilon calculated from U`*SIGMA*U
    :param alpha_scalar: Type I error rates
    :param powerwarn: vector of power calculation warning counts
    :return:
    """

    cdfpowercalc = ucdf[powercalc - 5]
    nue = total_N - rank_X

    if rank_U > nue and powercalc in (5, 8, 9):
        powerwarn.directfwarn(23)
        raise Exception("#TODO what kind of exception")

    if np.isnan(exeps) or nue <= 0:
        raise Exception("exeps is NaN or total_N  <= rank_X")

    undf1 = rank_C * rank_U
    undf2 = rank_U * nue

    # Create defaults - same for either SIGMA known or estimated
    sigstar = error_sum_square / nue
    q1 = np.trace(sigstar)
    q2 = np.trace(hypo_sum_square)
    q3 = q1 ** 2
    q4 = np.sum(np.power(sigstar, 2))
    q5 = np.trace(sigstar * hypo_sum_square)
    lambar = q1 / rank_U

    # Case 1
    # Enter loop to compute E1-E5 based on known SIGMA
    if sig_type == 0 and ip_plan == 0:
        epsn_num = q3 + q1 * q2 * 2 / rank_C
        epsn_den = q4 + q5 * 2 / rank_C
        epsn = epsn_num / (rank_U * epsn_den)
        e_1_2 = exeps
        e_4 = eps
        if cdfpowercalc == 1:
            e_3_5 = eps
        else:
            e_3_5 = epsn

    # Case 2
    # Enter loop to compute E1-E5 based on estimated SIGMA
    if sig_type == 1 and ip_plan == 0:
        nu_est = n_est - rank_est
        if nu_est <= 1:
            raise Exception("ERROR 81: Too few estimation df in LASTUNI. df = N_EST - RANK_EST <= 1.")

        # For POWERCALC =6=HF, =7=CM, =8=GG critical values
        epstilde_r = ((nu_est + 1) * q3 - 2 * q4) / (rank_U * (nu_est * q4 - q3))
        epstilde_r_min = min(epstilde_r, 1)
        mult = np.power(nu_est, 2) + nu_est - 2

        epsnhat_num = q3 * nu_est * (nu_est + 1) + q1 * q2 * 2 * mult / rank_C - q4 * 2 * nu_est
        epsnhat_den = q4 * nu_est * nu_est + q5 * 2 * mult / rank_C - q3 * nu_est
        epsnhat = epsnhat_num / (rank_U * epsnhat_den)

        nua0 = (nu_est - 1) + nu_est * (nu_est - 1) / 2
        tau10 = nu_est * ((nu_est + 1) * q1 * q1 - 2 * q4) / (nu_est * nu_est + nu_est - 2)
        tau20 = nu_est * (nu_est * q4 - q1 * q1) / (nu_est * nu_est + nu_est - 2)

        epsda = tau10 * (nua0 - 2) * (nua0 - 4) / (rank_U * nua0 * nua0 * tau20)
        epsda = max(min(epsda, 1), 1 / rank_U)
        epsna = (1 + 2 * (q2 / rank_C) / q1) / (1 / epsda + 2 * rank_U * (q5 / rank_C) / (q1 * q1))
        omegaua = q2 * epsna * (rank_U / q1)

        # Set E_1_2 for all tests

        # for UN or Box critical values
        if powercalc in (5, 9):
            e_1_2 = epsda

        # for HF crit val
        if powercalc == 6:
            if rank_U <= nue:
                e_1_2 = epstilde_r_min
            else:
                e_1_2 = epsda

        # for CM crit val
        if powercalc == 7:
            e_1_2 = epsda

        # for GG crit val
        if powercalc == 8:
            e_1_2 = eps

        # Set E_3_5 for all tests
        if cdfpowercalc == 1:
            e_3_5 = eps
        else:
            e_3_5 = epsnhat

        # Set E_4 for all tests
        e_4 = eps
        if powercalc == 7:
            e_4 = epsda

        # Compute DF for confidence limits for all tests
        cl1df = rank_U * nu_est * e_4 / e_3_5

    # case 3
    # Enter loop to compute E1-E5 when planning IP study
    if ip_plan == 1 and sig_type == 0:
        nu_ip = n_ip - rank_ip
        e_1_2 = exeps
        e_4 = eps

        if powercalc in (6, 7, 8):
            lambdap = np.concatenate((sigmastareval,
                                      np.power(sigmastareval, 2),
                                      np.power(sigmastareval, 3),
                                      np.power(sigmastareval, 4)), axis=1)
            sumlam = np.matrix(np.sum(lambdap, axis=0)).T
            kappa = np.multiply(np.multiply(np.matrix([[1], [2], [8], [48]]), nu_ip), sumlam)
            muprime2 = np.asscalar(kappa[1] + np.power(kappa[0], 2))
            meanq2 = np.asscalar(np.multiply(np.multiply(nu_ip, nu_ip + 1), sumlam[1])
                                 + np.multiply(nu_ip, np.sum(sigmastareval * sigmastareval.T)))

            et1 = muprime2 / np.power(nu_ip, 2)
            et2 = meanq2 / np.power(nu_ip, 2)
            ae_epsn_up = et1 + 2 * q1 * q2
            ae_epsn_dn = rank_U * (et2 + 2 * q5)
            aex_epsn = ae_epsn_up / ae_epsn_dn
            e_3_5 = aex_epsn
        else:
            epsn_num = q3 + q1 * q2 * 2 / rank_C
            epsn_den = q4 + q5 * 2 / rank_C
            epsn = epsn_num / (rank_U * epsn_den)
            e_3_5 = epsn

    # Error checking
    if e_1_2 < 1 / rank_U:
        e_1_2 = 1 / rank_U
        powerwarn.directfwarn(17)
    if e_1_2 > 1:
        e_1_2 = 1
        powerwarn.directfwarn(18)

    # Obtain noncentrality and critical value for power point estimate
    omega = e_3_5 * q2 / lambar
    if powercalc == 7 & sig_type == 1 & ip_plan == 0:
        omega = omegaua

    fcrit = finv(1 - alpha_scalar, undf1 * e_1_2, undf2 * e_1_2)

    # Compute power point estimate
    # 1. Muller, Edwards & Taylor 2002 CDF exact, Davies' algorithm
    if cdfpowercalc in (3, 4):
        df1 = float("nan")
        df2 = float("nan")
        fmethod = float("nan")
        accuracy = 10 ** (-round - 1)
        qweight = np.concatenate((sigmastareval, -sigmastareval * fcrit * undf1 / undf2))
        qnuvec = np.concatenate((np.full((rank_U, 1), rank_C), np.full((rank_U, 1), total_N - rank_X)), axis=0)
        dgover = np.diag(1 / np.sqrt(np.squeeze(np.asarray(sigmastareval))))
        factori = sigmastarevec * dgover
        omegstar = factori.T * hypo_sum_square * factori
        qnoncen = np.concatenate((np.diag(omegstar), np.zeros((rank_U, 1))), axis=0)
        # TODO cdfpowr = qprob()
        cdfpowr = float("nan")
        if np.isnan(cdfpowr):
            powerwarn.directfwarn(19)
        else:
            power = 1 - cdfpowr

    # 2. Muller, Edwards & Taylor 2002 and Muller Barton 1989 CDF approx
    # UCDFTEMP[]=4 reverts to UCDFTEMP[]=2 if exact CDF fails
    if cdfpowercalc in (1, 2) or (cdfpowercalc == 4 and np.isnan(power)):
        df1 = undf1 * e_3_5
        df2 = undf2 * e_4
        prob, fmethod = probf(fcrit, df1, df2, omega)
        powerwarn.fwarn(fmethod, 1)
        if fmethod == 4 and prob == 1:
            power = alpha_scalar
        else:
            power = 1 - prob

    # Compute CL for power, if requested by user
    #if cltype == 2:
        # change from chi sq to F, and only change:)
        #raise Exception("CLTYPE=2 for UNIREP awaiting implementation")

    if cltype == 1:
        if cdfpowercalc > 2:
            raise Exception("ERROR 82: Any use of Exact CDF is incompatible with computation of CL for power.")

        # Calculate lower bound for power
        if alpha_cl <= tolerance:
            prob_l = 1 - alpha_scalar
            fmethod_l = 5
            noncen_l = float('nan')
        else:
            chi_l = chi2.ppf(alpha_cl, cl1df)
            noncen_l = omega * (chi_l / cl1df)
            prob_l, fmethod_l = probf(fcrit, df1, df2, noncen_l)
            powerwarn.fwarn(fmethod_l, 2)

        if fmethod_l == 4 and prob_l == 1:
            power_l = alpha_scalar
        else:
            power_l = 1 - prob_l

        # Calculate upper bound for power
        if alpha_cu <= tolerance:
            prob_u = 0
            fmethod_u = 5
            noncen_u = float('nan')
        else:
            chi_u = chi2.ppf(1 - alpha_cu, cl1df)
            noncen_u = omega * (chi_u / cl1df)
            prob_u, fmethod_u = probf(fcrit, df1, df2, noncen_u)
            powerwarn.fwarn(fmethod_u, 2)

        if fmethod_u == 4 and prob_u == 1:
            power_u = alpha_scalar
        else:
            power_u = 1 - prob_u

        return power, power_l, power_u

    return power
