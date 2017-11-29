import numpy as np

from glmpowercalc.exceptions.ranksymm_validation_exception import RanksymmValidationException


def ranksymm(matrix, tolerance):
    """This function computes the rank of a square symmetric
           nonnegative definite matrix via eigenvalues.
                                                            
       :param matrix: Class define by Matrix(), with matrix and label
       :param tolerance:  value not tolerated, numeric zero (global)
                                                
       :return rankmatrix: 
           = . if MATRIX is not symmetric or positive definite
           = rank of the matrix
    """
    # empty matrix
    if np.shape(matrix)[1] == 0:
        raise RanksymmValidationException("ERROR 55: Matrix {0} does not exist.".format(matrix))

    # number of rows not equal to number of columns
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        raise RanksymmValidationException("ERROR 56: Matrix {0} is not square.".format(matrix))

    # matrix with all missing values
    if np.isnan(matrix).all():
        raise RanksymmValidationException("ERROR 57: Matrix {0} is all missing values.".format(matrix))

    maxabsval = abs(matrix).max()

    # matrix with all zero
    if maxabsval == 0:
        raise RanksymmValidationException("ERROR 58: Matrix {0} has MAX(ABS(all elements)) = exact zero.".format(matrix))

    nmatrix = matrix / maxabsval
    evals = np.linalg.eigvals(nmatrix)

    # matrix not symmetric
    if abs(nmatrix - nmatrix.T).max() >= tolerance ** 0.5:
        raise RanksymmValidationException("ERROR 59: Matrix {0} is not symmetric within sqrt(tolerance).".format(matrix))

    # matrix not non-negative definite
    if evals.min() < -tolerance ** 0.5:
        raise RanksymmValidationException("ERROR 60: Matrix {0} is *NOT* non-negative definite (and has at \
              least one eigenvalue strictly less than \
              zero). This may happen due to programming \
              error or rounding error of a nearly LTFR \
              matrix. This may be able to be fixed using \
              usual scaling/centering techniques. The \
              Eigenvalues/MAX(ABS(original matrix)) are: {1}. \
              The max(abs(original matrix)) is {2}.".format(matrix, evals, maxabsval))

    rankmatrix = sum(evals >= tolerance)
    return rankmatrix
