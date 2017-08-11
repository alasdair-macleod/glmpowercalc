import numpy as np
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf
from glmpowercalc.glmmpcl import glmmpcl

def hlt(rank_C, rank_U, rank_X, total_N, eval_HINVE, alphascalar, mmethod,
        optpowermat2, cltype, n_est, rank_est, alpha_cl, alpha_cu, tolerance, powerwarn):
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
    :param optpowermat2: options matrix specifying CL options
    :param cltype:
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
    # MMETHOD[0]  Choices for Hotelling-Lawley Trace
    #       = 1  Pillai (1954, 55) 1 moment null approx
    #       = 2  McKeon (1974) two moment null approx
    #       = 3  Pillai (1959) one moment null approx+ OS noncen mult
    #       = 4  McKeon (1974) two moment null approx+ OS noncen mult
    if mmethod[0] == 1 or mmethod[0] == 3:
        df2 = min_rank_C_U * (total_N - rank_X - rank_U - 1) + 2
    elif mmethod[0] == 2 or mmethod[0] == 4:
        nu_df2 = (total_N - rank_X)*(total_N - rank_X) - (total_N - rank_X)*(2*rank_U + 3) + rank_U*(rank_U + 3)
        de_df2 = (total_N - rank_X)*(rank_C + rank_U + 1) - (rank_C + 2*rank_U + rank_U*rank_U - 1)
        df2 = 4 + (rank_C*rank_U +2) * (nu_df2/de_df2)

    # df2 need to > 0 and eigenvalues not missing
    if df2 <= 0 or np.isnan(eval_HINVE):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
        if mmethod[0] > 2 or min_rank_C_U == 1:
            hlt = eval_HINVE * (total_N - rank_X) / total_N
            omega = (total_N * min_rank_C_U) * (hlt / min_rank_C_U)
        else:
            hlt = eval_HINVE
            omega = df2 * (hlt / min_rank_C_U)

        hlt_fcrit = finv(1 - alphascalar, df1, df2)
        hlt_prob, hlt_fmethod = probf(hlt_fcrit, df1, df2, omega)
        powerwarn.fwarn(hlt_fmethod, 1)

        if hlt_fmethod == 4 and hlt_prob == 1:
            power = alphascalar
        else:
            power = 1 - hlt_prob

    if cltype >= 1:
        if np.isnan(power):
            powerwarn.directfwarn(16)
        else:
            f_a = omega /df1
            return glmmpcl(f_a, alphascalar, df1, total_N, df2, cltype, n_est, rank_est,
                    alpha_cl, alpha_cu, tolerance, powerwarn)

    return power_l, power, power_u


