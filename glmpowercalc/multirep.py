import numpy as np

from glmpowercalc.constants import Constants
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf
from glmpowercalc.glmmpcl import glmmpcl


def hlt(rank_C, rank_U, rank_X, total_N, eval_HINVE, alphascalar, MultiHLT,
        cl_type, n_est, rank_est, alpha_cl, alpha_cu, tolerance, powerwarn):
    """
    This module calculates power for Hotelling-Lawley trace
    based on the Pillai F approximation. HLT is the "population value"
    Hotelling Lawley trace. F1 and DF2 are the hypothesis and
    error degrees of freedom, OMEGA is the non-centrality parameter, and
    FCRIT is the critical value from the F distribution.

    :param rank_C: rank of C matrix
    :param rank_U: rank of U matrix
    :param rank_X: rank of X matrix
    :param total_N: total N
    :param eval_HINVE: eigenvalues for H*INV(E)
    :param alphascalar: size of test
    :param mmethod: multirep method
    :param cl_type:
    :param n_est:
    :param rank_est:
    :param alpha_cl:
    :param alpha_cu:
    :param tolerance:
    :param powerwarn: calculation_state object
    :return: power, power for Hotelling-Lawley trace & CL if requested
    """
    min_rank_C_U = min(rank_C, rank_U)
    df1 = rank_C * rank_U

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 1  Pillai (1954, 55) 1 moment null approx
    #       = 2  McKeon (1974) two moment null approx
    #       = 3  Pillai (1959) one moment null approx+ OS noncen mult
    #       = 4  McKeon (1974) two moment null approx+ OS noncen mult
    if MultiHLT == Constants.MULTI_HLT_PILLAI or MultiHLT == Constants.MULTI_HLT_PILLAI_OS:
        df2 = min_rank_C_U * (total_N - rank_X - rank_U - 1) + 2
    elif MultiHLT == Constants.MULTI_HLT_MCKEON or MultiHLT == Constants.MULTI_HLT_MCKEON_OS:
        nu_df2 = (total_N - rank_X)*(total_N - rank_X) - (total_N - rank_X)*(2*rank_U + 3) + rank_U*(rank_U + 3)
        de_df2 = (total_N - rank_X)*(rank_C + rank_U + 1) - (rank_C + 2*rank_U + rank_U*rank_U - 1)
        df2 = 4 + (rank_C*rank_U + 2) * (nu_df2/de_df2)

    # df2 need to > 0 and eigenvalues not missing
    if df2 <= 0 or np.isnan(eval_HINVE[0]):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
        if (MultiHLT == Constants.MULTI_HLT_PILLAI_OS or
                MultiHLT == Constants.MULTI_HLT_MCKEON_OS) or min_rank_C_U == 1:
            hlt = eval_HINVE * (total_N - rank_X) / total_N
            omega = (total_N * min_rank_C_U) * (hlt / min_rank_C_U)
        else:
            hlt = eval_HINVE
            omega = df2 * (hlt / min_rank_C_U)

        power = multi_power(alphascalar, df1, df2, omega, powerwarn)

    power_l, power_u, fmethod_l, fmethod_u, noncen_l, noncen_u = glmmpcl(alphascalar, df1, total_N, df2, cl_type, n_est, rank_est,
                                                                            alpha_cl, alpha_cu, tolerance, powerwarn, power, omega)

    return power_l, power, power_u


def pbt(rank_C, rank_U, rank_X, total_N, eval_HINVE, alphascalar, MultiPBT,
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
    :param alphascalar: size of test
    :param MultiPBT: multirep method
    :param cl_type:
    :param n_est:
    :param rank_est:
    :param alpha_cl:
    :param alpha_cu:
    :param tolerance:
    :param powerwarn: calculation_state object
    :return: power, power for Pillai-Bartlett trace & CL if requested
    """

    # MMETHOD[1]  Choices for Pillai-Bartlett Trace
    #   = 1  Pillai (1954, 55) one moment null approx
    #   = 2  Muller (1998) two moment null approx
    #   = 3  Pillai (1959) one moment null approx + OS noncen mult
    #   = 4  Muller (1998) two moment null approx + OS noncen mult
    if MultiPBT == Constants.MULTI_PBT_PILLAI or MultiPBT == Constants.MULTI_PBT_PILLAI_OS:
        df1 = rank_C * rank_U
        df2 = min(rank_C, rank_U) * (total_N - rank_X + min(rank_C, rank_U) - rank_U)

    elif MultiPBT == Constants.MULTI_PBT_MULLER or MultiPBT == Constants.MULTI_PBT_MULLER_OS:
        mu1 = rank_C * rank_U / (total_N - rank_X + rank_C)
        factor1 = (total_N - rank_X + rank_C -rank_U) / (total_N - rank_X + rank_C - 1)
        factor2 = (total_N - rank_X) / (total_N - rank_X + rank_C + 2)
        variance = 2 * rank_C * rank_U * factor1 * factor2 / (total_N - rank_X + rank_C)**2
        mu2 = variance + mu1**2
        m1 = mu1 / min(rank_C, rank_U)
        m2 = mu2 / (min(rank_C, rank_U) * min(rank_C, rank_U))
        denom = m2 - m1 * m1
        df1 = 2 * m1 * (m1 - m2) / denom
        df2 = 2 * (m1 - m2) * (1 - m1) /denom

    if df2 <= 0 or np.isnan(eval_HINVE[0]):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
        if (MultiPBT == Constants.MULTI_PBT_PILLAI_OS or MultiPBT == Constants.MULTI_PBT_MULLER_OS)\
                or min(rank_U, rank_C) == 1:
            evalt = eval_HINVE * (total_N - rank_X) / total_N
        else:
            evalt = eval_HINVE

        v = sum(evalt / (np.ones((min(rank_C, rank_U), 1)) + evalt))

        if (min(rank_C, rank_U) - v) <= tolerance:
            power = float('nan')
        else:
            if (MultiPBT == Constants.MULTI_PBT_PILLAI_OS or MultiPBT == Constants.MULTI_PBT_MULLER_OS)\
                    or min(rank_U, rank_C) == 1:
                omega = total_N * min(rank_C, rank_U) * v / (min(rank_C, rank_U) - v)
            else:
                omega = df2 * v / (min(rank_C, rank_U) - v)

            power = multi_power(alphascalar, df1, df2, omega, powerwarn)

    power_l, power_u, fmethod_l, fmethod_u, noncen_l, noncen_u = glmmpcl(alphascalar, df1, total_N, df2, cl_type, n_est,
                                                                         rank_est,
                                                                         alpha_cl, alpha_cu, tolerance, powerwarn,
                                                                         power, omega)
    return power_l, power, power_u


def wlk(rank_C, rank_U, rank_X, total_N, eval_HINVE, alphascalar, MultiWLK,
        cl_type, n_est, rank_est, alpha_cl, alpha_cu, tolerance, powerwarn):
    """
    This module calculates power for Wilk's Lambda based on
    the F approx. method.  W is the "population value" of Wilks` Lambda,
    DF1 and DF2 are the hypothesis and error degrees of freedom, OMEGA
    is the noncentrality parameter, and FCRIT is the critical value
    from the F distribution. RM, RS, R1, and TEMP are intermediate
    variables.

    :param rank_C: rank of C matrix
    :param rank_U: rank of U matrix
    :param rank_X: rank of X matrix
    :param total_N: total N
    :param eval_HINVE: eigenvalues for H*INV(E)
    :param alphascalar: size of test
    :param MultiWLK: multirep method
    :param cl_type:
    :param n_est:
    :param rank_est:
    :param alpha_cl:
    :param alpha_cu:
    :param tolerance:
    :param powerwarn: calculation_state object
    :return: power, power for Hotelling-Lawley trace & CL if requested
    """
    min_rank_C_U = min(rank_C, rank_U)
    df1 = rank_C * rank_U

    # MMETHOD default= [4,2,2]
    # MMETHOD[2] Choices for Wilks' Lambda
    #       = 1  Rao (1951) two moment null approx
    #       = 2  Rao (1951) two moment null approx
    #       = 3  Rao (1951) two moment null approx + OS noncen mult
    #       = 4  Rao (1951) two moment null approx + OS noncen mult
    if np.isnan(eval_HINVE[0]):
        w = float('nan')
        powerwarn.directfwarn(15)
    else:
        if MultiWLK == Constants.MULTI_WLK_RAO or MultiWLK == Constants.MULTI_WLK_RAO_OS or min_rank_C_U == 1:
            w = np.exp(np.sum(-np.log(np.ones((min_rank_C_U, 1)) + eval_HINVE * (total_N - rank_X)/total_N)))
        else:
            w = np.exp(np.sum(-np.log(np.ones((min_rank_C_U, 1)) + eval_HINVE)))

    if min_rank_C_U == 1:
        df2 = total_N - rank_X -rank_U + 1
        rs = 1
        tempw = w
    else:
        rm = total_N - rank_X - (rank_U - rank_C + 1)/2
        rs = np.sqrt(rank_C*rank_C*rank_U*rank_U - 4) / (rank_C*rank_C + rank_U*rank_U - 5)
        r1 = (rank_U - rank_C - 2)/4
        if np.isnan(w):
            tempw = float('nan')
        else:
            tempw = np.power(w, 1/rs)
        df2 = (rm * rs) - 2 * r1

    if np.isnan(tempw):
        omega = float('nan')
    else:
        if MultiWLK == Constants.MULTI_WLK_RAO or MultiWLK == Constants.MULTI_WLK_RAO_OS or min_rank_C_U == 1:
            omega = (total_N * rs) * (1 - tempw) /tempw
        else:
            omega = df2 * (1 - tempw) / tempw

    if df2 <= 0 or np.isnan(w) or np.isnan(omega):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
        power = multi_power(alphascalar, df1, df2, omega, powerwarn)

    power_l, power_u, fmethod_l, fmethod_u, noncen_l, noncen_u = glmmpcl(alphascalar, df1, total_N, df2, cl_type, n_est,
                                                                         rank_est,
                                                                         alpha_cl, alpha_cu, tolerance, powerwarn,
                                                                         power, omega)
    return power_l, power, power_u



def special(rank_C, rank_U, rank_X, total_N, eval_HINVE, alphascalar,
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
    :param alphascalar: size of test
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
        power = multi_power(alphascalar, df1, df2, omega, powerwarn)

    power_l, power_u, fmethod_l, fmethod_u, noncen_l, noncen_u = glmmpcl(alphascalar, df1, total_N, df2, cl_type, n_est,
                                                                         rank_est,
                                                                         alpha_cl, alpha_cu, tolerance, powerwarn,
                                                                         power, omega)
    return power_l, power, power_u


def multi_power(alphascalar, df1, df2, omega, powerwarn):
    """ The common part for these four multirep methods
        Computing power """
    fcrit = finv(1 - alphascalar, df1, df2)
    prob, fmethod = probf(fcrit, df1, df2, omega)
    powerwarn.fwarn(fmethod, 1)
    if fmethod == Constants.FMETHOD_NORMAL_LR and prob == 1:
        power = alphascalar
    else:
        power = 1 - prob
    return power
