import numpy as np
from functools import reduce
import copy
import itertools


def uploy(factor_list):
    """
    This module creates a U contrast matrix with orthogonal polynomial coding for within subject factors.

    :param factor_list:list of levels of factors
    :return: U contrast matrix
    """

    return_list = dict()

    n_factor = len(factor_list)
    center_factor_list = list(map((lambda x: np.matrix(orpol((x-np.mean(x))/(np.sqrt(np.dot(x-np.mean(x), x-np.mean(x))))))), factor_list))
    zerotrend_list = list(map((lambda x: x[:, 0]), center_factor_list))
    highertrend_list = list(map((lambda x: x[:, 1:]), center_factor_list))

    u_grandmean = reduce((lambda x, y: np.kron(x, y)), zerotrend_list)
    #return_list['u_grandmean'] = u_grandmean

    u_maineffect = dict()
    for i in range(0, n_factor):
        temp_trend_list = copy.deepcopy(zerotrend_list)
        temp_trend_list[i] = highertrend_list[i]
        u_maineffect['f'+str(i)] = reduce((lambda x, y: np.kron(x, y)), temp_trend_list)
    return_list['u_maineffect'] = u_maineffect

    if n_factor >= 2:
        u_twoways = dict()
        for k in itertools.combinations(range(0, n_factor), 2):
            temp_trend_list = copy.deepcopy(zerotrend_list)
            temp_trend_list[k[0]] = highertrend_list[k[0]]
            temp_trend_list[k[1]] = highertrend_list[k[1]]
            u_twoways['f'+str(k)] = reduce((lambda x, y: np.kron(x, y)), temp_trend_list)
        return_list['u_twoways'] = u_twoways

    if n_factor >= 3:
        u_threeways = dict()
        for k in itertools.combinations(range(0, n_factor), 3):
            temp_trend_list = copy.deepcopy(zerotrend_list)
            temp_trend_list[k[0]] = highertrend_list[k[0]]
            temp_trend_list[k[1]] = highertrend_list[k[1]]
            temp_trend_list[k[2]] = highertrend_list[k[2]]
            u_threeways['f'+str(k)] = reduce((lambda x, y: np.kron(x, y)), temp_trend_list)
        return_list['u_threeways'] = u_threeways

    return return_list


def orpol(x, maxdegree=None, weights=None):
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
    assert x is not None
    n = np.shape(x)[0]

    # deal with the maxdegree
    if maxdegree is None:
        degree_desire = min(n, 19)
    else:
        assert maxdegree <= n, 'Please use a maxdegree smaller or equal to the number of value in x'
        degree_desire = min(maxdegree, 19)

    # deal with the weights
    if weights is None:
        weights = np.ones(n)
    assert np.shape(weights)[0] == n, 'Please specify weight for each value in x'

    orth_ploy = np.empty((degree_desire+1, n))
    orth_ploy[0, ] = np.zeros(n)
    qx = np.ones(n)

    for j in range(0, degree_desire):
        A = np.sqrt(np.sum(np.multiply(weights, np.power(qx, 2))))
        orth_ploy[j+1, ] = qx/A
        B = np.sum(np.multiply(np.multiply(x, weights), np.power(qx, 2)))/np.sum(np.multiply(weights, np.power(qx, 2)))
        qx = np.multiply(x-B, orth_ploy[j+1, ]) - A * orth_ploy[j, ]

    return orth_ploy[1:, ].T


def test():
    print(uploy([[1,2,3], [1,2,3]]))


if __name__ == '__main__':
    test()