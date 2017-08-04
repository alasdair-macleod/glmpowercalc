import numpy as np
from scipy.stats import chi2
from scipy import special
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf


def glmmpcl(f_a, alphatest, dfh, n2, dfe2, cltype, n_est, rank_est,
            alpha_cl, alpha_cu, tolerance, powerwarn):
    """
    This module computes confidence intervals for noncentrality and
    power for a General Linear Hypothesis (GLH  Ho:C*beta=theta0) test
    in the General Linear Univariate Model (GLUM: y=X*beta+e, HILE
    GAUSS), based on estimating the effect, error variance, or neither.
    Methods from Taylor and Muller (1995).

    :param f_a: = MSH/MSE, the F value observed if BETAhat=BETA and
                Sigmahat=Sigma, under the alternative hypothesis, with:
                    MSH=Mean Square Hypothesis (effect variance)
                    MSE=Mean Square Error (error variance)
                NOTE:
                    F_A = (N2/N1)*F_EST and
                    MSH = (N2/N1)*MSH_EST,
                    with "_EST" indicating value which was observed
                    in sample 1 (source of estimates)
    :param alphatest: Significance level for target GLUM test
    :param dfh: degrees of freedom for target GLH
    :param n2:
    :param dfe2: Error df for target hypothesis
    :param cltype:  =1 if Sigma estimated and Beta known
                    =2 if Sigma estimated and Beta estimated
    :param n_est: (scalar) # of observations in analysis which yielded
                    BETA and SIGMA estimates
    :param rank_est: (scalar) design matrix rank in analysis which
                        yielded BETA and SIGMA estimates
    :param alpha_cl: Lower tail probability for confidence interval
    :param alpha_cu: Upper tail probability for confidence interval
    :param tolerance:
    :param powerwarn: calculation_state object
    :return:
        power_l, power confidence interval lower bound
        power_u, power confidence interval upper bound
        fmethod_l, Method used to calculate probability from F CDF
                    used in lower confidence limits power calculation
        fmethod_u, Method used to calculate probability from F CDF
                    used in lower confidence limits power calculation
        noncen_l, noncentrality confidence interval lower bound
        noncen_u, noncentrality confidence interval upper bound
        powerwarn, vector of power calculation warning counts
    """

    # Calculate noncentrality
    dfe1 = n_est - rank_est
    noncen_e = dfh * f_a
    fcrit = finv(1-alphatest, dfh, dfe2)

    # Calculate lower bound for noncentrality
    if alpha_cl <= tolerance:
        noncen_l = 0
    elif cltype == 1:
        chi_l = chi2.ppf(alpha_cl, dfe1)
        noncen_l = (chi_l /dfe1) * noncen_e
    elif cltype == 2:
        bound_l = finv(1-alpha_cl, dfh, dfe1)
        if f_a <= bound_l:
            noncen_l = 0
        else:
            noncen_l = special.ncfdtrinc(dfh, dfe1, 1-alpha_cl, f_a)
            # the ncfdtrinc function seems always return a nan

    # Calculate lower bound for power
    if alpha_cl <= tolerance:
        prob = 1 - alphatest
        fmethod_l = 5
    else:
        prob, fmethod_l = probf(fcrit, dfh, dfe2, noncen_l)
        powerwarn.fwarn(fmethod_l, 2)

    if fmethod_l == 4 and prob == 1:
        power_l = alphatest
    else:
        power_l = 1 - prob

    # Calculate upper bound for noncentrality
    if alpha_cu <= tolerance:
        noncen_u = float('Inf')
    elif cltype == 1:
        chi_u = chi2.ppf(1 - alpha_cu, dfe1)
        noncen_u = (chi_u / dfe1) * noncen_e
    elif cltype == 2:
        bound_u = finv(alpha_cu, dfh, dfe1)
        if f_a <= bound_u:
            noncen_u = 0
        else:
            noncen_u = special.ncfdtrinc(dfh, dfe1, alpha_cu, f_a)

    # Calculate upper bound for power
    if alpha_cu <= tolerance:
        prob = 0
        fmethod_u = 5
    else:
        prob, fmethod_u = probf(fcrit, dfh, dfe2, noncen_u)
        powerwarn.fwarn(fmethod_u, 3)

    if fmethod_u == 4 and prob == 1:
        power_u = alphatest
    else:
        power_u = 1 - prob

    # warning for conservative confidence interval
    if cltype > 1 and n2 != n_est:
        if alpha_cl > 0 and noncen_l == 0:
            powerwarn.directfwarn(5)
        if alpha_cl == 0 and noncen_u == 0:
            powerwarn.directfwarn(10)

    return power_l, power_u, fmethod_l, fmethod_u, noncen_l, noncen_u
