import numpy as np
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf
from glmpowercalc.glmmpcl import glmmpcl

def wlk(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha_scalar, m_method,
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
    :param alpha_scalar: size of test
    :param m_method: multirep method
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
        if m_method[2] == 2 or m_method[2] == 4 or min_rank_C_U == 1:
            w = np.exp(np.sum(-np.log(np.ones((min_rank_C_U, 1)) + eval_HINVE * (total_N - rank_X)/total_N)))
        else:
            w = np.exp(np.sum(-np.log(np.ones((min_rank_C_U, 1)) + eval_HINVE)))

    if min_rank_C_U == 1:
        df2 = total_N - rank_X -rank_U + 1
        rs = 1
        tempw = w
    else:
        rm = total_N - rank_X - (rank_U - rank_C + 1) / 2
        rs = np.sqrt((rank_C*rank_C*rank_U*rank_U - 4) / (rank_C*rank_C + rank_U*rank_U - 5))
        r1 = (rank_U * rank_C - 2)/4
        if np.isnan(w):
            tempw = float('nan')
        else:
            tempw = np.power(w, 1/rs)
        df2 = (rm * rs) - 2 * r1

    if np.isnan(tempw):
        omega = float('nan')
    else:
        if m_method[2] == 2 or m_method[2] == 4 or min_rank_C_U == 1:
            omega = (total_N * rs) * (1 - tempw) /tempw
        else:
            omega = df2 * (1 - tempw) / tempw

    if df2 <= 0 or np.isnan(w) or np.isnan(omega):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
        wlk_fcrit = finv(1 - alpha_scalar, df1, df2)
        wlk_prob, wlk_fmethod = probf(wlk_fcrit, df1, df2, omega)
        powerwarn.fwarn(wlk_fmethod, 1)

        if wlk_fmethod == 4 and wlk_prob == 1:
            power = alpha_scalar
        else:
            power = 1 - wlk_prob

    if cl_type >= 1:
        if np.isnan(power):
            powerwarn.directfwarn(16)
        else:
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


