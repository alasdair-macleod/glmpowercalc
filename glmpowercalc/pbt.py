import numpy as np
from glmpowercalc.finv import finv
from glmpowercalc.probf import probf
from glmpowercalc.glmmpcl import glmmpcl

def pbt(rank_C, rank_U, rank_X, total_N, eval_HINVE, alphascalar, mmethod,
        optpowermat2, cltype, n_est, rank_est, alpha_cl, alpha_cu, tolerance, powerwarn):
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
    :param mmethod: multirep method
    :param optpowermat2: options matrix specifying CL options
    :param cltype:
    :param n_est:
    :param rank_est:
    :param alpha_cl:
    :param alpha_cu:
    :param tolerance:
    :param powerwarn: calculation_state object
    :return: power, power for Pillai-Bartlett trace & CL if requested
    """

    fmethod = 0


    # MMETHOD[1]  Choices for Pillai-Bartlett Trace
    #   = 1  Pillai (1954, 55) one moment null approx
    #   = 2  Muller (1998) two moment null approx
    #   = 3  Pillai (1959) one moment null approx + OS noncen mult
    #   = 4  Muller (1998) two moment null approx + OS noncen mult
    if mmethod[1] == 1 or mmethod[1] == 3:
        df1 = rank_C * rank_U
        df2 = min(rank_C, rank_U) * (total_N - rank_X + min(rank_C, rank_U) - rank_U)

    elif mmethod[1] == 2 or mmethod ==4:
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

    if df2 <= 0 or np.isnan(eval_HINVE[0, 0]):
        power = float('nan')
        powerwarn.directfwarn(15)
    else:
