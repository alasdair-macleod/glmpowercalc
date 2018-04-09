from unittest import TestCase
from glmpowercalc.orpol import orpol, uploy
import numpy as np


class TestOrpol(TestCase):
    def test_orpol1(self):
        """
        This should return the expected value
        test case is from Emerson 1968 article, and the expected value is from SAS/IML output
        """
        expected = [[0.25, -0.472456, 0.433013, -0.163663],
                    [0.25, -0.094491, -0.144338, 0.272772],
                    [0.25, 0.0944911, -0.144338, -0.272772],
                    [0.25, 0.4724556, 0.433013, 0.163663]]
        actual = orpol([0, 2, 3, 5], 4, [2, 6, 6, 2])
        result = np.round(actual, 6)
        self.assertTrue((expected == result).all)

    def test_orpol2(self):
        """
        Test for maxdegree < n
        This should return the expected value
        """
        expected = [[0.25, -0.472456, 0.433013],
                    [0.25, -0.094491, -0.144338],
                    [0.25, 0.0944911, -0.144338],
                    [0.25, 0.4724556, 0.433013]]
        actual = orpol([0, 2, 3, 5], 3, [2, 6, 6, 2])
        result = np.round(actual, 6)
        self.assertTrue((expected == result).all)

    def test_orpol_none_maxdegree(self):
        """
        Test for maxdegree not specify
        This should return the expected value
        """
        expected = [[0.25, -0.472456, 0.433013, -0.163663],
                    [0.25, -0.094491, -0.144338, 0.272772],
                    [0.25, 0.0944911, -0.144338, -0.272772],
                    [0.25, 0.4724556, 0.433013, 0.163663]]
        actual = orpol(x=[0, 2, 3, 5], weights=[2, 6, 6, 2])
        result = np.round(actual, 6)
        self.assertTrue((expected == result).all)

    def test_uploy_onefactor(self):
        """

        :return:
        """
        exp_u_maineffect = [[-0.707107, 0.408248],
                            [0, -0.816497],
                            [0.707107, 0.408248]]
        actual = uploy([[1, 2, 3]])
        self.assertTrue((exp_u_maineffect == np.round(actual['u_maineffect']['f0'], 6)).all)

    def test_uploy_twofactors(self):
        """

        :return:
        """
        exp_u_maineffect_f0 = [[-0.408248, 0.235702],
                               [-0.408248, 0.235702],
                               [-0.408248, 0.235702],
                               [0, -0.471405],
                               [0, -0.471405],
                               [0, -0.471405],
                               [0.408248, 0.235702],
                               [0.408248, 0.235702],
                               [0.408248, 0.235702]]
        exp_u_maineffect_f1 = [[-0.408248, 0.235702],
                               [0, -0.471405],
                               [0.408248, 0.235702],
                               [-0.408248, 0.235702],
                               [0, -0.471405],
                               [0.408248, 0.235702],
                               [-0.408248, 0.235702],
                               [0, -0.471405],
                               [0.408248, 0.235702]]
        exp_u_twoways_f01 = [[0.5, -0.288675, -0.288675, 0.166667],
                             [0, 0.577350, 0, -0.333333],
                             [-0.5, -0.288675, 0.288675, 0.166667],
                             [0, 0, 0.577350, -0.333333],
                             [0, 0, 0, 0.666667],
                             [0, 0, -0.57735, -0.333333],
                             [-0.5, 0.288675, -0.288675, 0.166667],
                             [0, -0.57735, 0, -0.333333],
                             [0.5, 0.288675, 0.2886751, 0.166667]]
        actual = uploy([[1, 2, 3], [1, 2, 3]])
        self.assertTrue((exp_u_maineffect_f0 == np.round(actual['u_maineffect']['f0'], 6)).all)
        self.assertTrue((exp_u_maineffect_f1 == np.round(actual['u_maineffect']['f1'], 6)).all)
        self.assertTrue((exp_u_twoways_f01 == np.round(actual['u_twoways']['f(0, 1)'], 6)).all)

    def test_uploy_threefactors(self):
        exp_u_maineffect_f0 = [[-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828]]
        exp_u_maineffect_f1 = [[-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [-0.235702, 0.1360828],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0, -0.272166],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828],
                               [0.2357023, 0.1360828]]
        exp_u_maineffect_f2 = [[-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647],
                               [-0.211667, 0.1710883],
                               [-0.042333, -0.268853],
                               [0.2540003, 0.0977647]]
        exp_u_twoways_f01 = [[0.2886751, -0.166667, -0.166667, 0.096225],
                             [0.2886751, -0.166667, -0.166667, 0.096225],
                             [0.2886751, -0.166667, -0.166667, 0.096225],
                             [0, 0.3333333, 0, -0.19245],
                             [0, 0.3333333, 0, -0.19245],
                             [0, 0.3333333, 0, -0.19245],
                             [-0.288675, -0.166667, 0.1666667, 0.096225],
                             [-0.288675, -0.166667, 0.1666667, 0.096225],
                             [-0.288675, -0.166667, 0.1666667, 0.096225],
                             [0, 0, 0.3333333, -0.19245],
                             [0, 0, 0.3333333, -0.19245],
                             [0, 0, 0.3333333, -0.19245],
                             [0, 0, 0, 0.3849002],
                             [0, 0, 0, 0.3849002],
                             [0, 0, 0, 0.3849002],
                             [0, 0, -0.333333, -0.19245],
                             [0, 0, -0.333333, -0.19245],
                             [0, 0, -0.333333, -0.19245],
                             [-0.288675, 0.1666667, -0.166667, 0.096225],
                             [-0.288675, 0.1666667, -0.166667, 0.096225],
                             [-0.288675, 0.1666667, -0.166667, 0.096225],
                             [0, -0.333333, 0, -0.19245],
                             [0, -0.333333, 0, -0.19245],
                             [0, -0.333333, 0, -0.19245],
                             [0.2886751, 0.1666667, 0.1666667, 0.096225],
                             [0.2886751, 0.1666667, 0.1666667, 0.096225],
                             [0.2886751, 0.1666667, 0.1666667, 0.096225]]
        exp_u_twoways_f02 = [[0.2592379, -0.20954, -0.149671, 0.1209777],
                             [0.0518476, 0.3292764, -0.029934, -0.190108],
                             [-0.311086, -0.119737, 0.1796053, 0.0691301],
                             [0.2592379, -0.20954, -0.149671, 0.1209777],
                             [0.0518476, 0.3292764, -0.029934, -0.190108],
                             [-0.311086, -0.119737, 0.1796053, 0.0691301],
                             [0.2592379, -0.20954, -0.149671, 0.1209777],
                             [0.0518476, 0.3292764, -0.029934, -0.190108],
                             [-0.311086, -0.119737, 0.1796053, 0.0691301],
                             [0, 0, 0.2993422, -0.241955],
                             [0, 0, 0.0598684, 0.3802156],
                             [0, 0, -0.359211, -0.13826],
                             [0, 0, 0.2993422, -0.241955],
                             [0, 0, 0.0598684, 0.3802156],
                             [0, 0, -0.359211, -0.13826],
                             [0, 0, 0.2993422, -0.241955],
                             [0, 0, 0.0598684, 0.3802156],
                             [0, 0, -0.359211, -0.13826],
                             [-0.259238, 0.2095395, -0.149671, 0.1209777],
                             [-0.051848, -0.329276, -0.029934, -0.190108],
                             [0.3110855, 0.1197369, 0.1796053, 0.0691301],
                             [-0.259238, 0.2095395, -0.149671, 0.1209777],
                             [-0.051848, -0.329276, -0.029934, -0.190108],
                             [0.3110855, 0.1197369, 0.1796053, 0.0691301],
                             [-0.259238, 0.2095395, -0.149671, 0.1209777],
                             [-0.051848, -0.329276, -0.029934, -0.190108],
                             [0.3110855, 0.1197369, 0.1796053, 0.0691301]]
        exp_u_twoways_f12 = [[0.2592379, -0.20954, -0.149671, 0.1209777],
                             [0.0518476, 0.3292764, -0.029934, -0.190108],
                             [-0.311086, -0.119737, 0.1796053, 0.0691301],
                             [0, 0, 0.2993422, -0.241955],
                             [0, 0, 0.0598684, 0.3802156],
                             [0, 0, -0.359211, -0.13826],
                             [-0.259238, 0.2095395, -0.149671, 0.1209777],
                             [-0.051848, -0.329276, -0.029934, -0.190108],
                             [0.3110855, 0.1197369, 0.1796053, 0.0691301],
                             [0.2592379, -0.20954, -0.149671, 0.1209777],
                             [0.0518476, 0.3292764, -0.029934, -0.190108],
                             [-0.311086, -0.119737, 0.1796053, 0.0691301],
                             [0, 0, 0.2993422, -0.241955],
                             [0, 0, 0.0598684, 0.3802156],
                             [0, 0, -0.359211, -0.13826],
                             [-0.259238, 0.2095395, -0.149671, 0.1209777],
                             [-0.051848, -0.329276, -0.029934, -0.190108],
                             [0.3110855, 0.1197369, 0.1796053, 0.0691301],
                             [0.2592379, -0.20954, -0.149671, 0.1209777],
                             [0.0518476, 0.3292764, -0.029934, -0.190108],
                             [-0.311086, -0.119737, 0.1796053, 0.0691301],
                             [0, 0, 0.2993422, -0.241955],
                             [0, 0, 0.0598684, 0.3802156],
                             [0, 0, -0.359211, -0.13826],
                             [-0.259238, 0.2095395, -0.149671, 0.1209777],
                             [-0.051848, -0.329276, -0.029934, -0.190108],
                             [0.3110855, 0.1197369, 0.1796053, 0.0691301]]
        exp_u_threeways_f012 = [[-0.3175, 0.2566325, 0.1833089, -0.148167, 0.1833089, -0.148167, -0.105833, 0.0855442],
                                [-0.0635, -0.40328, 0.0366618, 0.2328336, 0.0366618, 0.2328336, -0.021167, -0.134427],
                                [0.3810004, 0.1466471, -0.219971, -0.084667, -0.219971, -0.084667, 0.1270001, 0.0488824],
                                [0, 0, -0.366618, 0.2963336, 0, 0, 0.2116669, -0.171088],
                                [0, 0, -0.073324, -0.465667, 0, 0, 0.0423334, 0.268853],
                                [0, 0, 0.4399413, 0.1693335, 0, 0, -0.254, -0.097765],
                                [0.3175003, -0.256632, 0.1833089, -0.148167, -0.183309, 0.1481668, -0.105833, 0.0855442],
                                [0.0635001, 0.4032796, 0.0366618, 0.2328336, -0.036662, -0.232834, -0.021167, -0.134427],
                                [-0.381, -0.146647, -0.219971, -0.084667, 0.2199707, 0.0846668, 0.1270001, 0.0488824],
                                [0, 0, 0, 0, -0.366618, 0.2963336, 0.2116669, -0.171088],
                                [0, 0, 0, 0, -0.073324, -0.465667, 0.0423334, 0.268853],
                                [0, 0, 0, 0, 0.4399413, 0.1693335, -0.254, -0.097765],
                                [0, 0, 0, 0, 0, 0, -0.423334, 0.3421766],
                                [0, 0, 0, 0, 0, 0, -0.084667, -0.537706],
                                [0, 0, 0, 0, 0, 0, 0.5080005, 0.1955295],
                                [0, 0, 0, 0, 0.3666178, -0.296334, 0.2116669, -0.171088],
                                [0, 0, 0, 0, 0.0733236, 0.4656671, 0.0423334, 0.268853],
                                [0, 0, 0, 0, -0.439941, -0.169334, -0.254, -0.097765],
                                [0.3175003, -0.256632, -0.183309, 0.1481668, 0.1833089, -0.148167, -0.105833, 0.0855442],
                                [0.0635001, 0.4032796, -0.036662, -0.232834, 0.0366618, 0.2328336, -0.021167, -0.134427],
                                [-0.381, -0.146647, 0.2199707, 0.0846668, -0.219971, -0.084667, 0.1270001, 0.0488824],
                                [0, 0, 0.3666178, -0.296334, 0, 0, 0.2116669, -0.171088],
                                [0, 0, 0.0733236, 0.4656671, 0, 0, 0.0423334, 0.268853],
                                [0, 0, -0.439941, -0.169334, 0, 0, -0.254, -0.097765],
                                [-0.3175, 0.2566325, -0.183309, 0.1481668, -0.183309, 0.1481668, -0.105833, 0.0855442],
                                [-0.0635, -0.40328, -0.036662, -0.232834, -0.036662, -0.232834, -0.021167, -0.134427],
                                [0.3810004, 0.1466471, 0.2199707, 0.0846668, 0.2199707, 0.0846668, 0.1270001, 0.0488824]]
        actual = uploy([[1, 2, 3], [1, 3, 5], [1, 5, 12]])
        self.assertTrue((exp_u_maineffect_f0 == np.round(actual['u_maineffect']['f0'], 6)).all)
        self.assertTrue((exp_u_maineffect_f1 == np.round(actual['u_maineffect']['f1'], 6)).all)
        self.assertTrue((exp_u_maineffect_f2 == np.round(actual['u_maineffect']['f2'], 6)).all)
        self.assertTrue((exp_u_twoways_f01 == np.round(actual['u_twoways']['f(0, 1)'], 6)).all)
        self.assertTrue((exp_u_twoways_f02 == np.round(actual['u_twoways']['f(0, 2)'], 6)).all)
        self.assertTrue((exp_u_twoways_f12 == np.round(actual['u_twoways']['f(1, 2)'], 6)).all)
        self.assertTrue((exp_u_threeways_f012 == np.round(actual['u_threeways']['f(0, 1, 2)'], 6)).all)
