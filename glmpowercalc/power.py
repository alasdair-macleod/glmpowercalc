import numpy as np
import warnings
from glmpowercalc.constants import Constants
from glmpowercalc.ranksymm import ranksymm


class CL:
    def __init__(self, cl_desire=False, sigma_type=False, beta_type=False,
                 n_est=None, rank_est=None, alpha_cl=0.025, alpha_cu=0.025):
        """

        :param cl_desire: confidence intervals is desired or not
        :param sigma_type: sigma is estimated or not
        :param beta_type: beta is estimated or not
        :param n_est: number of observations in analysis which yielded beata and sigma estimates
        :param rank_est: rank of design matrix in analysis which yielded beta and sigma estimates
        :param alpha_cl: lower tail probability for power CL
        :param alpha_cu: upper tail probability for power CL

        :return cl_type: CLTYPE_DESIRED_ESTIMATE, CLTYPE_DESIRED_KNOWN, CLTYPE_NOT_DESIRED
        """
        self.sigma_type = sigma_type

        if sigma_type:  # sigma is estimated
            if cl_desire:  # sigma is desired
                if beta_type:  # beta is estimated
                    self.cl_type = Constants.CLTYPE_DESIRED_ESTIMATE
                else:  # beta is known
                    self.cl_type = Constants.CLTYPE_DESIRED_KNOWN
            else:
                self.cl_type = Constants.CLTYPE_NOT_DESIRED

            assert n_est is not None
            assert rank_est is not None
            self.n_est = n_est
            self.rank_est = rank_est

            if alpha_cl < 0 or \
                alpha_cu < 0 or \
                alpha_cl >= 1 or \
                alpha_cu >= 1 or \
                    (alpha_cl + alpha_cu >= 1):
                raise Exception('ERROR 35: ALPHA_CL and ALPHA_CU must both be >= 0 and <= 1.')
            self.alpha_cl = alpha_cl
            self.alpha_cu = alpha_cu

        else:
            raise Exception('sigma_type need to be estimated to calculate CL')


class IP:
    def __init__(self, ip_plan=False, n_ip=None, rank_ip=None):
        """

        :param ip_plan: indicates whether power is computed within the context of planning an interval pilot
        :param n_ip: of observations planned for the internal pilot of the future study (required if IP_PLAN=True)
        :param rank_ip: rank of the design matrix used in the future study (required if IP_PLAN=True)
        """
        self.ip_plan = ip_plan
        if ip_plan:
            assert n_ip is not None
            assert rank_ip is not None

            if n_ip <= rank_ip:
                raise Exception('ERROR 90: N_IP must > RANK_IP')

            self.n_ip = n_ip
            self.rank_ip = rank_ip


class Power:
    def __init__(self):

        self.c_matrix = np.matrix([[1]])
        self.beta = np.matrix([[1]])
        self.sigma = np.matrix([[2]])
        self.essencex = np.matrix([[1]])
        self.u_matrix = np.matrix(np.ones(np.shape(self.beta)[1]))
        self.theta = np.zeros((np.shape(self.c_matrix)[0], np.shape(self.u_matrix)[1]))
        self.repn = np.matrix([[10]])
        self.betascal = np.matrix([[0.5]])
        self.sigscal = np.matrix([[1]])
        self.rhoscal = np.matrix([[1]])
        self.alpha = np.matrix([[0.05]])
        self.round = 3
        self.tolerance = 1e-12
        self.UnirepUncorrected = Constants.UCDF_MULLER2004_APPROXIMATION
        self.UnirepHuynhFeldt = Constants.UCDF_MULLER2004_APPROXIMATION
        self.UnirepHuynhFeldtChiMuller = Constants.UCDF_MULLER2004_APPROXIMATION
        self.UnirepGeisserGreenhouse = Constants.UCDF_MULLER2004_APPROXIMATION
        self.UnirepBox = Constants.UCDF_MULLER2004_APPROXIMATION
        self.EpsilonAppHuynhFeldt = Constants.EPSILON_MULLER2004
        self.EpsilonAppHuynhFeldtChiMuller = Constants.EPSILON_MULLER2004
        self.EpsilonAppGeisserGreenhouse = Constants.EPSILON_MULLER2004
        self.MultirepHLT = Constants.MULTI_HLT_MCKEON_OS
        self.MultiPBT = Constants.MULTI_PBT_MULLER
        self.MultiWLK = Constants.MULTI_WLK_RAO
        self.opt_noncencl = False
        self.opt_calc_collapse = False
        self.opt_calc_hlt = True
        self.opt_calc_pbt = True
        self.opt_calc_wlk = True
        self.opt_calc_un = False
        self.opt_calc_hf = False
        self.opt_calc_cm = False
        self.opt_calc_gg = True
        self.opt_calc_box = False
        self.opt_fracrepn = False

        self.CL = CL()
        self.IP = IP()

    def check(self):

        if self.CL.cl_type == Constants.CLTYPE_NOT_DESIRED and self.opt_noncencl:
            raise Exception("ERROR 83: NONCENCL is not a valid option when CL not desired.")

        #TODO the whole opt stuff
        self.option_check(min_rank_c_u, num_col_u, num_row_c)

        #Check repn
        if self.repn.min() <= self.tolerance:
            raise Exception('ERROR 10: All REPN values must be > TOLERANCE > 0.')

        #TODO need to verify the logic
        if self.opt_fracrepn and self.repn.dtype == 'float':
            raise Exception('ERROR 11: All REPN values must be positive integers. To allow fractional REPN values, '
                            'specify opt_fracrepn')

        #Check sigscal
        if self.sigscal.min() <= self.tolerance:
            raise Exception('ERROR 12: All SIGSCAL values must be > TOLERANCE > 0.')

        #Check alpha
        if self.alpha.min() <= self.tolerance or self.alpha.max() >= 1:
            raise Exception('ERROR 13: All ALPHA values must be > TOLERANCE > 0 and < 1.')

        #Check round
        if self.round < 1 or self.round > 15:
            raise Exception('ERROR 15: User specified ROUND < 1 OR ROUND >15')

        #Check tolerance
        if self.tolerance <= 0:
            raise Exception('ERROR 17: User specified TOLERANCE <= zero.')
        if self.tolerance >= 0.01:
            raise Exception('WARNING 6: User specified TOLERANCE >= 0.01. This is the value assumed to be numeric '
                            'zero and affects many calculations. Please check that this value is correct.')

        #Check UCDF
        if self.UnirepUncorrected == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        self.UnirepHuynhFeldt == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        self.UnirepHuynhFeldtChiMuller == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        self.UnirepGeisserGreenhouse == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        self.UnirepBox == Constants.UCDF_MULLER1989_APPROXIMATION:
            warnings.warn('WARNING 7: You have chosen the Muller, Barton (1989) approximation for the UNIREP '
                          'statistic CDF. Muller, Edwards, Taylor (2004) found NO condition where their approximation '
                          'was not superior to this Muller, Barton approximation.  Suggest specifying '
                          'UCDF_MULLER2004_APPROXIMATION; '
                          'unless you are performing a backwards comparison calculation.')

        #Check IP_PLAN and SIGTYPE
        if self.IP.ip_plan and self.CL.sigma_type:
            raise Exception('ERROR 91: SIGMA must be known when planning an internal pilot.')


        # define initial parameters for power calculation
        num_col_x = np.shape(self.essencex)[1]  # Q
        num_row_c = np.shape(self.c_matrix)[0]  # A
        num_col_u = np.shape(self.u_matrix)[1]  # B
        num_response = np.shape(self.beta)[1]  # P

        #Check on size and conformity of matrices
        if np.diag(self.sigma).min() < -self.tolerance:
            raise Exception('At least one variance < -tolerance. Check SIGMA')
        if np.shape(self.u_matrix)[0] != np.shape(self.sigma)[0]:
            raise Exception('ERROR 37: The number of rows of U is not equal to number of rows of SIGMA.')
        if np.shape(self.c_matrix)[1] != np.shape(self.beta)[0]:
            raise Exception('ERROR 38: The number of columns of C is not equal to number of rows of BETA.')
        if np.shape(self.beta)[1] != np.shape(self.u_matrix)[0]:
            raise Exception('ERROR 39: The number of columns of BETA is not equal to number of rows of U.')

        # Q
        if num_col_x != np.shape(self.beta)[0]:
            raise Exception('ERROR 40: The number of columns of ESSENCEX is not equal to number of rows of BETA.')

        # A
        if num_row_c > np.shape(self.c_matrix)[1]:
            raise Exception('ERROR 41: The number of rows of C is greater than the number of columns of C.')

        # B
        if num_col_u > np.shape(self.u_matrix)[0]:
            raise Exception('ERROR 42: The number of columns of U is greater than the number of rows of U.')

        if np.shape(self.theta)[0] != num_row_c or \
            np.shape(self.theta)[1] != num_col_u:
            raise Exception('ERROR 43: The THETA0 matrix does not conform to CBETAU.')

        if self.sigscal.min() <= self.tolerance:
            raise Exception('ERROR 44: Smallest value in SIGSCAL <= TOLERANCE (too small)')

        xpx = (self.essencex.T * self.essencex + (self.essencex.T * self.essencex).T) / 2
        cpc = (self.c_matrix * self.c_matrix.T + (self.c_matrix * self.c_matrix.T).T) / 2
        upu = (self.u_matrix.T * self.u_matrix + (self.u_matrix.T * self.u_matrix).T) / 2
        rank_X = ranksymm(xpx, self.tolerance)  # R
        rank_c = ranksymm(cpc, self.tolerance)
        rank_u = ranksymm(upu, self.tolerance)
        min_rank_c_u = min(rank_c, rank_u)  # S
        xpxginv = np.linalg.pinv(xpx) # (Moore-Penrose) pseudo-inverse, the same method in IML--GINV()
        m_matrix = self.c_matrix * xpxginv * self.c_matrix.T
        rank_m = ranksymm(m_matrix)

        cbetau = self.c_matrix * self.beta * self.u_matrix
        sigma_diag_inv = np.linalg.inv(np.diag(np.sqrt(np.diag(self.sigma))))
        rho = sigma_diag_inv * self.sigma * sigma_diag_inv

        # Check and warning for less than full rank ESSENCEX
        # R
        if 









    def option_check(self, min_rank_c_u, num_col_u, num_row_c):
        if not (self.opt_calc_collapse |
                    self.opt_calc_hlt |
                    self.opt_calc_pbt |
                    self.opt_calc_wlk |
                    self.opt_calc_un |
                    self.opt_calc_hf |
                    self.opt_calc_cm |
                    self.opt_calc_gg |
                    self.opt_calc_box):
            raise Exception("ERROR 9: No power calculation selected.")
        if self.opt_calc_collapse:
            if min_rank_c_u > 1:
                self.opt_calc_collapse = False
                raise Exception("WARNING 1: Rank(C*BETA*U) > 1, so COLLAPSE option ignored.")
            if num_col_u == 1:
                self.opt_calc_hlt = 0
                self.opt_calc_pbt = 0
                self.opt_calc_wlk = 0
                self.opt_calc_un = 0
                self.opt_calc_hf = 0
                self.opt_calc_cm = 0
                self.opt_calc_gg = 0
                self.opt_calc_box = 0
                raise Exception(
                    "WARNING 2: B = 1, so that all tests coincide (univariate case). "
                    "Since collapse option is on, power is given as one value with the heading POWER. "
                    "To print powers with a heading for  each test, specify collapse option off.")
            if num_col_u > 1 and num_row_c == 1:
                if self.opt_calc_hlt == 0 and \
                                self.opt_calc_pbt == 0 and \
                                self.opt_calc_wlk == 0:
                    pass

    def power(self):
        pass
