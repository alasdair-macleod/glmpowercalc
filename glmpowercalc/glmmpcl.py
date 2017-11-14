import numpy as np
from scipy.stats import chi2
from scipy import special

from glmpowercalc.constants import Constants
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf


def glmmpcl(alphatest, dfh, n2, dfe2, cl_type, n_est, rank_est,
            alpha_cl, alpha_cu, tolerance, powerwarn, power, omega):
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
    :param cl_type:  =1 if Sigma estimated and Beta known
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
    if cl_type == Constants.CLTYPE_DESIRED_KNOWN | cl_type == Constants.CLTYPE_DESIRED_ESTIMATE:
        if np.isnan(power):
            powerwarn.directfwarn(16)
        else:
            f_a = omega / dfh
            dfe1, fcrit, noncen_e = calc_noncentrality(alphatest, dfe2, dfh, f_a, n_est, rank_est)
            noncen_l = lowerbound_noncentrality(alpha_cl, cl_type, dfe1, dfh, f_a, noncen_e, tolerance)
            fmethod_l, power_l = lowerbound_power(alpha_cl, alphatest, dfe2, dfh, fcrit, noncen_l, powerwarn, tolerance)
            noncen_u = upperbound_noncentrality(alpha_cu, cl_type, dfe1, dfh, f_a, noncen_e, tolerance)
            fmethod_u, power_u = upperbound_power(alpha_cu, alphatest, dfe2, dfh, fcrit, noncen_u, powerwarn, tolerance)

            warn_conservative_ci(alpha_cl, cl_type, n2, n_est, noncen_l, noncen_u, powerwarn)

            return power_l, power_u, fmethod_l, fmethod_u, noncen_l, noncen_u


def warn_conservative_ci(alpha_cl, cl_type, n2, n_est, noncen_l, noncen_u, powerwarn):
    """warning for conservative confidence interval"""
    if (cl_type == Constants.CLTYPE_DESIRED_KNOWN |
            cl_type == Constants.CLTYPE_DESIRED_ESTIMATE) and n2 != n_est:
        if alpha_cl > 0 and noncen_l == 0:
            powerwarn.directfwarn(5)
        if alpha_cl == 0 and noncen_u == 0:
            powerwarn.directfwarn(10)


def upperbound_power(alpha_cu, alphatest, dfe2, dfh, fcrit, noncen_u, powerwarn, tolerance):
    """Calculate upper bound for power"""
    if alpha_cu <= tolerance:
        prob = 0
        fmethod_u = Constants.FMETHOD_MISSING
    else:
        prob, fmethod_u = probf(fcrit, dfh, dfe2, noncen_u)
        powerwarn.fwarn(fmethod_u, 3)
    if fmethod_u == Constants.FMETHOD_NORMAL_LR and prob == 1:
        power_u = alphatest
    else:
        power_u = 1 - prob
    return fmethod_u, power_u


def lowerbound_power(alpha_cl, alphatest, dfe2, dfh, fcrit, noncen_l, powerwarn, tolerance):
    """Calculate lower bound for power"""
    if alpha_cl <= tolerance:
        prob = 1 - alphatest
        fmethod_l = Constants.FMETHOD_MISSING
    else:
        prob, fmethod_l = probf(fcrit, dfh, dfe2, noncen_l)
        powerwarn.fwarn(fmethod_l, 2)
    if fmethod_l == Constants.FMETHOD_NORMAL_LR and prob == 1:
        power_l = alphatest
    else:
        power_l = 1 - prob
    return fmethod_l, power_l


def upperbound_noncentrality(alpha_cu, cl_type, dfe1, dfh, f_a, noncen_e, tolerance):
    """Calculate upper bound for noncentrality"""
    if alpha_cu <= tolerance:
        noncen_u = float('Inf')
    elif cl_type == Constants.CLTYPE_DESIRED_KNOWN:
        chi_u = chi2.ppf(1 - alpha_cu, dfe1)
        noncen_u = (chi_u / dfe1) * noncen_e
    elif cl_type == Constants.CLTYPE_DESIRED_ESTIMATE:
        bound_u = finv(alpha_cu, dfh, dfe1)
        if f_a <= bound_u:
            noncen_u = 0
        else:
            noncen_u = special.ncfdtrinc(dfh, dfe1, alpha_cu, f_a)
    return noncen_u


def lowerbound_noncentrality(alpha_cl, cl_type, dfe1, dfh, f_a, noncen_e, tolerance):
    """Calculate lower bound for noncentrality"""
    if alpha_cl <= tolerance:
        noncen_l = 0
    elif cl_type == Constants.CLTYPE_DESIRED_KNOWN:
        chi_l = chi2.ppf(alpha_cl, dfe1)
        noncen_l = (chi_l / dfe1) * noncen_e
    elif cl_type == Constants.CLTYPE_DESIRED_ESTIMATE:
        bound_l = finv(1 - alpha_cl, dfh, dfe1)
        if f_a <= bound_l:
            noncen_l = 0
        else:
            noncen_l = special.ncfdtrinc(dfh, dfe1, 1 - alpha_cl, f_a)
            # the ncfdtrinc function seems always return a nan
    return noncen_l


def calc_noncentrality(alphatest, dfe2, dfh, f_a, n_est, rank_est):
    """Calculate noncentrality"""
    dfe1 = n_est - rank_est
    noncen_e = dfh * f_a
    fcrit = finv(1 - alphatest, dfh, dfe2)
    return dfe1, fcrit, noncen_e
