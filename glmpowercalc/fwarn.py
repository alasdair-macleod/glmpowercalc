import numpy as np

powerwarn = np.zeros(23, dtype=np.int)

def fwarn(fmethod, cl):
    """
    This module updates the appropriate warning counts in POWERWARN based
    on the value of FMETHOD produced by the PROBF module (used to
    calculate powers).

    :param fmethod:
    :param cl:
        =1 if calculation of power of a tests
        =2 if calculation of lower CL for power of a tests
        =3 if calculation of upper CL for power of a tests
    :return: powerwarn, Vector of power calculation warning counts
    """
    powerwarn[(fmethod-1)+5*(cl-1) - 1] = powerwarn[(fmethod-1)+5*(cl-1) - 1] + 1

    return powerwarn


class WarningCount