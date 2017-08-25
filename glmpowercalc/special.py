import numpy as np
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf
from glmpowercalc.glmmpcl import glmmpcl


def special(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha_scalar,
            cl_type, n_est, rank_est, alpha_cl, alpha_cu, tolerance, powerwarn):
    """
    This module performs two disparate tasks. For B=1 (UNIVARIATE
    TEST), the powers are calculated more efficiently. For A=1 (SPECIAL
    MULTIVARIATE CASE), exact multivariate powers are calculated.
    Powers for the univariate tests require separate treatment.
    DF1 & DF2 are the hypothesis and error degrees of freedom,
    OMEGA is the noncentrality parameter, and FCRIT is the critical
    value from the F distribution.

    :param rank_C: rank of C matrix
    :param rank_U: rank of U matrix
    :param rank_X: rank of X matrix
    :param total_N: total N
    :param eval_HINVE: eigenvalues for H*INV(E)
    :param alpha_scalar: size of test
    :param cl_type:
    :param n_est:
    :param rank_est:
    :param alpha_cl:
    :param alpha_cu:
    :param tolerance:
    :param powerwarn: calculation_state object
    :return: power, power for Hotelling-Lawley trace & CL if requested
    """
    df1 = rank_C * rank_U
    df2 = total_N - rank_X - rank_U + 1

    if df2 <= 0 or np.isnan(eval_HINVE[0]):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
        omega = eval_HINVE[0] * (total_N - rank_X)
        special_fcrit = finv(1 - alpha_scalar, df1, df2)
        special_prob, special_fmethod = probf(special_fcrit, df1, df2, omega)
        powerwarn.fwarn(special_fmethod, 1)

        if special_fmethod == 4 and special_prob == 1:
            power = alpha_scalar
        else:
            power = 1 - special_prob

    if cl_type >= 1:
        if np.isnan(power):
            powerwarn.directfwarn(16)
        else:
            f_a = omega / df1
            power_l, power_u, fmethod_l, fmethod_u, noncen_l, noncen_u = glmmpcl(f_a,
                                                                                 alpha_scalar,
                                                                                 df1,
                                                                                 total_N,
                                                                                 df2,
                                                                                 cl_type,
                                                                                 n_est,
                                                                                 rank_est,
                                                                                 alpha_cl,
                                                                                 alpha_cu,
                                                                                 tolerance,
                                                                                 powerwarn)

    return power_l, power, power_u
