import numpy as np


class CalculationState(object):
    """object to hold stateful information for powerlib calculation"""

    powerwarn = None
    tolerance = None

    def __init__(self, tolerance):
        self.powerwarn = np.zeros(23, dtype=np.int)
        self.tolerance = tolerance

    def fwarn(self, fmethod, cl):
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
        self.powerwarn[(fmethod - 1) + 5 * (cl - 1) - 1] = self.powerwarn[(fmethod - 1) + 5 * (cl - 1) - 1] + 1

    def directfwarn(self, sequence):
        """
        This method is used for directly counting fwarn
        :param sequence: the sequence number of the warning
        :return: powerwarn
        """
        self.powerwarn[sequence - 1] = self.powerwarn[sequence - 1] + 1