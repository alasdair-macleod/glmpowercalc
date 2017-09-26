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

    #Get eigenvalues of covariance matrix associated with E. This is NOT
    #the USUAL sigma. This cov matrix is that of (Y-YHAT)*U, not of (Y-YHAT).
    #The covariance matrix is normalized to minimize numerical problems
    esig = sigmastar/ np.trace(sigmastar)
    seigval = np.linalg.eigvals(esig)
    slam1 = np.sum(seigval)**2
    slam2 = np.sum(np.square(seigval))
    slam3 = np.sum(seigval)
    eps = slam1 / (rank_U * slam2)

    deigval_low_high_order, mtp = np.unique(seigval, return_counts=True)
    deigval = deigval_low_high_order[::-1]
    d = len(deigval)

    return d, mtp, eps, deigval, slam1, slam2, slam3


def hfexeps(sigmastar, rank_U, tolerance, total_N, rank_X, u_methodtemp):
    """

    Univariate, HF STEP 2:
    This function computes the approximate expected value of
    the Huynh-Feldt estimate.

      FK  = 1st deriv of FNCT of eigenvalues
      FKK = 2nd deriv of FNCT of eigenvalues
      For HF, FNCT is epsilon tilde

    :param sigmastar:
    :param rank_U:
    :param tolerance:
    :param total_N:
    :param rank_X:
    :param u_methodtemp:
    :return:
    """
    d, mtp, eps, deigval, slam1, slam2, slam3 = firstuni(sigmastar=sigmastar,
                                                         rank_U=rank_U)

    h1 = total_N * slam1 - 2 * slam2
    h2 = (total_N - rank_X) * slam2 - slam1

