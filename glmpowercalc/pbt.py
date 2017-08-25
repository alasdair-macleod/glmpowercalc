import numpy as np
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf
from glmpowercalc.glmmpcl import glmmpcl

def pbt(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha_scalar, m_method,
        cl_type, n_est, rank_est, alpha_cl, alpha_cu, tolerance, powerwarn):
    """
    This module calculates power for Pillai-Bartlett trace based on the
    F approx. method.  V is the "population value" of PBT,
    DF1 and DF2 are the hypothesis and error degrees of freedom,
    OMEGA is the noncentrality parameter, and FCRIT is the
    critical value from the F distribution.

    :param rank_C: rank of C matrix
    :param rank_U: rank of U matrix
    :param rank_X: rank of X matrix
    :param total_N: total N
    :param eval_HINVE: eigenvalues for H*INV(E)
    :param alpha_scalar: size of test
    :param m_method: multirep method
    :param cl_type:
    :param n_est:
    :param rank_est:
    :param alpha_cl:
    :param alpha_cu:
    :param tolerance:
    :param powerwarn: calculation_state object
    :return: power, power for Pillai-Bartlett trace & CL if requested
    """

    min_rank_C_U = min(rank_C, rank_U)


    # MMETHOD[1]  Choices for Pillai-Bartlett Trace
    #   = 1  Pillai (1954, 55) one moment null approx
    #   = 2  Muller (1998) two moment null approx
    #   = 3  Pillai (1959) one moment null approx + OS noncen mult
    #   = 4  Muller (1998) two moment null approx + OS noncen mult
    if m_method[1] == 1 or m_method[1] == 3:
        df1 = rank_C * rank_U
        df2 = min_rank_C_U * (total_N - rank_X + min_rank_C_U - rank_U)

    elif m_method[1] == 2 or m_method ==4:
        mu1 = rank_C * rank_U / (total_N - rank_X + rank_C)
        factor1 = (total_N - rank_X + rank_C -rank_U) / (total_N - rank_X + rank_C - 1)
        factor2 = (total_N - rank_X) / (total_N - rank_X + rank_C + 2)
        variance = 2 * rank_C * rank_U * factor1 * factor2 / (total_N - rank_X + rank_C)**2
        mu2 = variance + mu1**2
        m1 = mu1 / min_rank_C_U
        m2 = mu2 / (min_rank_C_U * min_rank_C_U)
        denom = m2 - m1 * m1
        df1 = 2 * m1 * (m1 - m2) / denom
        df2 = 2 * (m1 - m2) * (1 - m1) /denom

    if df2 <= 0 or np.isnan(eval_HINVE[0]):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
        if m_method[1] > 2 or min(rank_U, rank_C) == 1:
            evalt = eval_HINVE * (total_N - rank_X) / total_N
        else:
            evalt = eval_HINVE

        v = sum(evalt / (np.ones((min_rank_C_U, 1)) + evalt))

        if (min_rank_C_U - v) <= tolerance:
            power = float('nan')
        else:
            if m_method[1] > 2 or min_rank_C_U == 1:
                omega = total_N * min_rank_C_U * v / (min_rank_C_U - v)
            else:
                omega = df2 * v / (min_rank_C_U - v)

            pbt_fcrit = finv(1 - alpha_scalar, df1, df2)
            pbt_prob, pbt_fmethod = probf(pbt_fcrit, df1, df2, omega)
            powerwarn.fwarn(pbt_fmethod, 1)

            if pbt_fmethod == 4 and pbt_prob == 1:
                power = alpha_scalar
            else:
                power = 1 - pbt_prob

    if cl_type >= 1 and not np.isnan(power):
        f_a = omega /df1
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