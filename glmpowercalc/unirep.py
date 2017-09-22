import numpy as np

def firstuni(sigmastar, rank_U, tolerance):
    """
    This module produces matrices required for Geisser-Greenhouse,
    Huynh-Feldt or uncorrected repeated measures power calculations. It
    is the first step. Program uses approximations of expected values of
    epsilon estimates due to Muller (1985), based on theorem of Fujikoshi
    (1978). Program requires that U be orthonormal and orthogonal to a
    columns of 1's.

    :param sigmastar: U` * (SIGMA # SIGSCALTEMP) * U
    :param rank_U: rank of U matrix
    :param tolerance:

    :return:
        D, number of distinct eigenvalues
        MTP, multiplicities of eigenvalues
        EPS, epsilon calculated from U`*SIGMA*U
        DEIGVAL, first eigenvalue
        SLAM1, sum of eigenvalues squared
        SLAM2, sum of squared eigenvalues
        SLAM3, sum of eigenvalues
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

    # Decide which eigenvalues are distinct
    d = 1
    mtp = [1]
    deigval = seigval[0]
    for cnt in range(2, rank_U):
        if deigval[d-1] - seigval[cnt-1] > tolerance:
            d = d + 1
            deigval = np.append(deigval, seigval[cnt-1])
            mtp = np.append(mtp, 1)
        else:
            mtp[d-1] = mtp[d-1] + 1

    return d, mtp, eps, deigval, slam1, slam2, slam3







