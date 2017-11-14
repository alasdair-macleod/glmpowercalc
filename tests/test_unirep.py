from unittest import TestCase
import numpy as np
from glmpowercalc import unirep
from glmpowercalc.calculation_state import CalculationState


class TestUnirep(TestCase):
    def test_Firstuni1(self):
        """ The value of the object should equal to expected """
        expected = (2,
                    np.array([1, 1]),
                    0.4310345,
                    np.array([-0.074456, 1.0744563]),
                    1,
                    1.16,
                    1)
        actual = unirep.firstuni(sigmastar=np.matrix([[1, 2], [3, 4]]),
                                 rank_U=2)
        self.assertEqual(actual[0], expected[0])
        self.assertTrue((actual[1] == expected[1]).all())
        self.assertAlmostEqual(actual[2], expected[2], places=7)
        self.assertAlmostEqual(actual[3][0], expected[3][0], delta=0.000001)
        self.assertAlmostEqual(actual[3][1], expected[3][1], delta=0.000001)
        self.assertAlmostEqual(actual[4], expected[4], places=7)
        self.assertAlmostEqual(actual[5], expected[5], places=7)
        self.assertAlmostEqual(actual[6], expected[6], places=7)

    def test_Firstuni2(self):
        """ should raise error """
        with self.assertRaises(Exception):
            actual = unirep.firstuni(sigmastar=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                     rank_U=2)

    def test_hfexeps(self):
        """ should return expected value """
        expected = 0.2901679
        actual = unirep.hfexeps(sigmastar=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5,
                                UnirepUncorrected=2)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_cmexeps(self):
        """ should return expected value """
        expected = 0.2757015
        actual = unirep.cmexeps(sigmastar=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5,
                                UnirepUncorrected=2)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_ggexeps(self):
        """ should return expected value """
        expected = 0.2975125
        actual = unirep.ggexeps(sigmastar=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5,
                                UnirepHuynhFeldt=2)
        self.assertAlmostEqual(actual, expected, delta=0.0000001)

    def test_lastuni1(self):
        """ case 1: should return expected value """
        expected = 0.9634686
        ucdf = [2, 2, 2, 2, 2]
        powercalc = 6
        rank_C = 1
        rank_U = 4
        rank_X = 1
        total_N = 20
        hypo_sum_square = np.matrix([[0.3125, 0.625, -0.625, 0.3125],
                                        [0.625, 1.25, -1.25, 0.625],
                                        [-0.625, -1.25, 1.25, -0.625],
                                        [0.3125, 0.625, -0.625, 0.3125]])
        error_sum_square = np.matrix([[4.47545, -3.3e-17, 1.055e-15, 1.648e-17],
                                        [-3.3e-17, 3.25337, -8.24e-18, 5.624e-16],
                                        [1.055e-15, -8.24e-18, 1.05659, -3.19e-17],
                                        [1.648e-17, 5.624e-16, -3.19e-17, 0.89699]])
        sig_type = 0
        ip_plan = 0
        rank_ip = 1
        n_ip = 10
        n_est = 10
        rank_est = float('nan')
        sigmastareval = np.matrix([[0.23555], [0.17123], [0.05561], [0.04721]])
        sigmastarevec = np.matrix([[-1, 4.51e-17, -2.01e-16, -4.61e-18],
                                    [2.776e-17, 1, -3.33e-16, -2.39e-16],
                                    [-2.74e-16, 2.632e-16, 1, 2.001e-16],
                                    [-4.61e-18, 2.387e-16, -2e-16, 1]])
        cltype = -1
        alpha_cl = 0.025
        alpha_cu = 0.025
        tolerance = 1e-12
        round = 2
        exeps = 0.7203684
        eps = 0.7203684
        alpha_scalar = 0.04
        powerwarn = CalculationState(1e-12)
        actual = unirep.lastuni(powercalc, rank_U, total_N, rank_X,
                                error_sum_square, hypo_sum_square, sig_type, ip_plan, rank_ip,
                                n_est, rank_est, n_ip, sigmastareval, sigmastarevec,
                                cltype, alpha_cl, alpha_cu, tolerance, round,
                                exeps, eps, alpha_scalar, powerwarn)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_lastuni2(self):
        """ case 2: should return expected value """
        expected = 0.9802819
        ucdf = [2, 2, 2, 2, 2]
        powercalc = 6
        rank_C = 1
        rank_U = 4
        rank_X = 1
        total_N = 20
        hypo_sum_square = np.matrix([[0.3125, 0.625, -0.625, 0.3125],
                                        [0.625, 1.25, -1.25, 0.625],
                                        [-0.625, -1.25, 1.25, -0.625],
                                        [0.3125, 0.625, -0.625, 0.3125]])
        error_sum_square = np.matrix([[4.47545, -3.3e-17, 1.055e-15, 1.648e-17],
                                        [-3.3e-17, 3.25337, -8.24e-18, 5.624e-16],
                                        [1.055e-15, -8.24e-18, 1.05659, -3.19e-17],
                                        [1.648e-17, 5.624e-16, -3.19e-17, 0.89699]])
        sig_type = 1
        ip_plan = 0
        rank_ip = 1
        n_ip = 10
        n_est = 10
        rank_est = 1
        sigmastareval = np.matrix([[0.23555], [0.17123], [0.05561], [0.04721]])
        sigmastarevec = np.matrix([[-1, 4.51e-17, -2.01e-16, -4.61e-18],
                                    [2.776e-17, 1, -3.33e-16, -2.39e-16],
                                    [-2.74e-16, 2.632e-16, 1, 2.001e-16],
                                    [-4.61e-18, 2.387e-16, -2e-16, 1]])
        cltype = -1
        alpha_cl = 0.025
        alpha_cu = 0.025
        tolerance = 1e-12
        round = 2
        exeps = 0.7203684
        eps = 0.7203684
        alpha_scalar = 0.04
        powerwarn = CalculationState(1e-12)
        actual = unirep.lastuni(powercalc, rank_U, total_N, rank_X,
                                error_sum_square, hypo_sum_square, sig_type, ip_plan, rank_ip,
                                n_est, rank_est, n_ip, sigmastareval, sigmastarevec,
                                cltype, alpha_cl, alpha_cu, tolerance, round,
                                exeps, eps, alpha_scalar, powerwarn)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_lastuni3(self):
        """ case 3: should return expected value """
        expected = 0.9605244
        ucdf = [2, 2, 2, 2, 2]
        powercalc = 6
        rank_C = 1
        rank_U = 4
        rank_X = 1
        total_N = 20
        hypo_sum_square = np.matrix([[0.3125, 0.625, -0.625, 0.3125],
                                        [0.625, 1.25, -1.25, 0.625],
                                        [-0.625, -1.25, 1.25, -0.625],
                                        [0.3125, 0.625, -0.625, 0.3125]])
        error_sum_square = np.matrix([[4.47545, -3.3e-17, 1.055e-15, 1.648e-17],
                                        [-3.3e-17, 3.25337, -8.24e-18, 5.624e-16],
                                        [1.055e-15, -8.24e-18, 1.05659, -3.19e-17],
                                        [1.648e-17, 5.624e-16, -3.19e-17, 0.89699]])
        sig_type = 0
        ip_plan = 1
        rank_ip = 1
        n_ip = 10
        n_est = 10
        rank_est = float('nan')
        sigmastareval = np.matrix([[0.23555], [0.17123], [0.05561], [0.04721]])
        sigmastarevec = np.matrix([[-1, 4.51e-17, -2.01e-16, -4.61e-18],
                                    [2.776e-17, 1, -3.33e-16, -2.39e-16],
                                    [-2.74e-16, 2.632e-16, 1, 2.001e-16],
                                    [-4.61e-18, 2.387e-16, -2e-16, 1]])
        cltype = -1
        alpha_cl = 0.025
        alpha_cu = 0.025
        tolerance = 1e-12
        round = 2
        exeps = 0.7203684
        eps = 0.7203684
        alpha_scalar = 0.04
        powerwarn = CalculationState(1e-12)
        actual = unirep.lastuni(powercalc, rank_U, total_N, rank_X,
                                error_sum_square, hypo_sum_square, sig_type, ip_plan, rank_ip,
                                n_est, rank_est, n_ip, sigmastareval, sigmastarevec,
                                cltype, alpha_cl, alpha_cu, tolerance, round,
                                exeps, eps, alpha_scalar, powerwarn)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_lastuni4(self):
        """ cltype=1: should return expected value """
        expected = (0.9802819, 0.7798431, 0.9996234)
        ucdf = [2, 2, 2, 2, 2]
        powercalc = 6
        rank_C = 1
        rank_U = 4
        rank_X = 1
        total_N = 20
        hypo_sum_square = np.matrix([[0.3125, 0.625, -0.625, 0.3125],
                                        [0.625, 1.25, -1.25, 0.625],
                                        [-0.625, -1.25, 1.25, -0.625],
                                        [0.3125, 0.625, -0.625, 0.3125]])
        error_sum_square = np.matrix([[4.47545, -3.3e-17, 1.055e-15, 1.648e-17],
                                        [-3.3e-17, 3.25337, -8.24e-18, 5.624e-16],
                                        [1.055e-15, -8.24e-18, 1.05659, -3.19e-17],
                                        [1.648e-17, 5.624e-16, -3.19e-17, 0.89699]])
        sig_type = 1
        ip_plan = 0
        rank_ip = 1
        n_ip = 10
        n_est = 10
        rank_est = 1
        sigmastareval = np.matrix([[0.23555], [0.17123], [0.05561], [0.04721]])
        sigmastarevec = np.matrix([[-1, 4.51e-17, -2.01e-16, -4.61e-18],
                                    [2.776e-17, 1, -3.33e-16, -2.39e-16],
                                    [-2.74e-16, 2.632e-16, 1, 2.001e-16],
                                    [-4.61e-18, 2.387e-16, -2e-16, 1]])
        cltype = 1
        alpha_cl = 0.025
        alpha_cu = 0.025
        tolerance = 1e-12
        round = 2
        exeps = 0.7203684
        eps = 0.7203684
        alpha_scalar = 0.04
        powerwarn = CalculationState(1e-12)
        actual = unirep.lastuni(powercalc, rank_U, total_N, rank_X,
                                error_sum_square, hypo_sum_square, sig_type, ip_plan, rank_ip,
                                n_est, rank_est, n_ip, sigmastareval, sigmastarevec,
                                cltype, alpha_cl, alpha_cu, tolerance, round,
                                exeps, eps, alpha_scalar, powerwarn)
        self.assertAlmostEqual(actual[0], expected[0], places=5)
        self.assertAlmostEqual(actual[1], expected[1], places=5)
        self.assertAlmostEqual(actual[2], expected[2], places=5)