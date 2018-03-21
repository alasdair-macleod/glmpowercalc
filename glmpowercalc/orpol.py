import numpy as np

def orpol(x, maxdegree, weights):
    """
    The orpol function generates orthogonal polynomials on a discrete set of points.
    Reference: Emerson 1968

    :param x: is an n*1 vector of values on which the polynomials are to be defined.
    :param maxdegree: specifies the maximum degree polynomial to be computed.
                        If maxdegree is omitted, the default value is min(n, 19).
    :param weights: specifies an n*1 vector of nonnegative weights associated with the points in x.
    :return: a matrix with n rows and maxdegree+1 columns
    """

    # deal with x
    if x is None:
        raise Exception('Please enter the vector of values on which the polynomials are to be defined')
    else:
        n = np.shape(x)[0]

    # deal with the maxdegree
    if maxdegree is None:
        degree_desire = min(n, 19)
    elif maxdegree > n:
        raise Exception('Please use a maxdegree smaller or equal to the number of value in x')
    else:
        degree_desire = min(maxdegree, 19)

    # deal with the weights
    if weights is None:
        weights = np.empty(n)
    elif np.shape(weights)[0] != n:
        raise Exception('Please specify weight for each value in x')

    orth_ploy = np.empty((degree_desire+1, n))
    orth_ploy[0,] = np.zeros(n)
    qx = np.ones(n)

    for j in range(0, degree_desire):
        A = np.sqrt(np.sum(np.multiply(weights, np.power(qx, 2))))
        orth_ploy[j+1, ] = qx/A
        B = np.sum(np.multiply(np.multiply(x, weights), np.power(qx, 2)))/np.sum(np.multiply(weights, np.power(qx, 2)))
        qx = np.multiply(x-B, orth_ploy[j+1, ]) - A * orth_ploy[j, ]

    return orth_ploy[1:, ]