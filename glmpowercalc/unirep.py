import numpy as np


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
    mtp     = np.matrix(mtp_array).T

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
    der2h1 = np.full((d, 1), 2*total_N-4)
    der2h2 = np.full((d, 1), 2*(total_N-rank_X)-2)
    fkk = (np.multiply(-derh1, derh2) / h2 + der2h1 - np.multiply(derh1, derh2) / h2 + 2 * h1 * np.power(derh2, 2) / h2 ** 2 - h1 * der2h2 / h2) / (h2 * rank_U)
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
        sum2 = sum(tm3)

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
    e1epshf = num01 /den01

    # u_method
    # =1 --> Muller and Barton (1989) approximation
    # =2 --> Method 1, Muller, Edwards, and Taylor (2004)
    if u_method == 1:
        exeps = e0epshf
    elif u_method == 2:
        exeps = e1epshf

    return exeps

