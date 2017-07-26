import numpy as np


def hlt(rank_C, rank_U, rank_X, total_N, eval_HINVE, alaphascalar, mmethod,
        optpowermat2, cltype, n_est, rank_est, alpha_cl, aloha_cu):
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
    :param alaphascalar: size of test
    :param mmethod: multirep method
    :param optpowermat2: options matrix specifying CL options
    :param cltype:
    :param n_est:
    :param rank_est:
    :param alpha_cl:
    :param aloha_cu:
    :return: power
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
        df2 = min_rank_C_U * (total_N - rank_X - rank_U) + 2
    elif mmethod[0] == 2 or mmethod[0] == 4:
        nu_df2 = (total_N - rank_X)*(total_N - rank_X) - (total_N - rank_X)*(2*rank_U + 3) + rank_U*(rank_U + 3)
        de_df2 = (total_N - rank_X)*(rank_C + rank_U + 1) - (rank_C + 2*rank_U + rank_U*rank_U - 1)
        df2 = 4 + (rank_C*rank_U +2) * (nu_df2/de_df2)

    # df2 need to > 0 and eigenvalues not missing
    if df2 <= 0 or np.isnan(eval_HINVE[0, 0]):
        pass
    else:
        if mmethod[0] > 2 or min_rank_C_U == 1:
            hlt = (eval_HINVE * (total_N - rank_X) / total_N).sum(axis=1)
            omega = (total_N * min_rank_C_U) * (hlt / min_rank_C_U)
        else:
            hlt = eval.sum(axis=1)
            omega = df2 * (hlt / min_rank_C_U)

        finv()
        probf()
        fwarn()

        if fmethod == 4 and prob == 1:
            power = alaphascalar
        else:
            power = 1 - prob





