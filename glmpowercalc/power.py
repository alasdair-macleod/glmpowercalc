import numpy as np
import warnings

from glmpowercalc import unirep, multirep
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

        if cl_desire:  # sigma is desired
            if sigma_type:  # sigma is estimated
                if beta_type:  # beta is estimated
                    self.cl_type = Constants.CLTYPE_DESIRED_ESTIMATE
                else:  # beta is known
                    self.cl_type = Constants.CLTYPE_DESIRED_KNOWN

                assert n_est is not None
                assert rank_est is not None

            else:
                raise Exception('sigma_type need to be estimated to calculate CL')

        else:
            self.cl_type = Constants.CLTYPE_NOT_DESIRED

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
        self.u_matrix = np.matrix(np.identity(np.shape(self.beta)[1]))
        self.theta_zero = np.zeros((np.shape(self.c_matrix)[0], np.shape(self.u_matrix)[1]))
        self.repn = np.matrix([[10]])
        self.rep_n = 10
        self.betascal = np.matrix([[0.5]])
        self.beta_scalar = 0.5
        self.sigscal = np.matrix([[1]])
        self.sigma_scalar = 1
        self.rhoscal = np.matrix([[1]])
        self.rho_scalar = 1
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
        self.MultiHLT = Constants.MULTI_HLT_MCKEON_OS
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
        self.opt_orthu = True
        self.opt_uniforce = False
        self.opt_cmwarn = True

        self.CL = CL()
        self.IP = IP()

    def power(self):

        if self.CL.cl_type == Constants.CLTYPE_NOT_DESIRED and self.opt_noncencl:
            raise Exception("ERROR 83: NONCENCL is not a valid option when CL not desired.")

        # Check repn
        if self.repn.min() <= self.tolerance:
            raise Exception('ERROR 10: All REPN values must be > TOLERANCE > 0.')

        # TODO need to verify the logic
        if self.opt_fracrepn and self.repn.dtype == 'float':
            raise Exception('ERROR 11: All REPN values must be positive integers. To allow fractional REPN values, '
                            'specify opt_fracrepn')

        # Check sigscal
        if self.sigscal.min() <= self.tolerance:
            raise Exception('ERROR 12: All SIGSCAL values must be > TOLERANCE > 0.')

        # Check alpha
        if self.alpha.min() <= self.tolerance or self.alpha.max() >= 1:
            raise Exception('ERROR 13: All ALPHA values must be > TOLERANCE > 0 and < 1.')

        # Check round
        if self.round < 1 or self.round > 15:
            raise Exception('ERROR 15: User specified ROUND < 1 OR ROUND >15')

        # Check tolerance
        if self.tolerance <= 0:
            raise Exception('ERROR 17: User specified TOLERANCE <= zero.')
        if self.tolerance >= 0.01:
            raise Exception('WARNING 6: User specified TOLERANCE >= 0.01. This is the value assumed to be numeric '
                            'zero and affects many calculations. Please check that this value is correct.')

        # Check UCDF
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

        # Check IP_PLAN and SIGTYPE
        if self.IP.ip_plan and self.CL.sigma_type:
            raise Exception('ERROR 91: SIGMA must be known when planning an internal pilot.')

        # define initial parameters for power calculation
        num_col_x = np.shape(self.essencex)[1]  # Q
        num_row_c = np.shape(self.c_matrix)[0]  # A
        num_col_u = np.shape(self.u_matrix)[1]  # B
        num_response = np.shape(self.beta)[1]  # P

        # Check on size and conformity of matrices
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

        if np.shape(self.theta_zero)[0] != num_row_c or \
                        np.shape(self.theta_zero)[1] != num_col_u:
            raise Exception('ERROR 43: The THETA0 matrix does not conform to CBETAU.')

        if self.sigscal.min() <= self.tolerance:
            raise Exception('ERROR 44: Smallest value in SIGSCAL <= TOLERANCE (too small)')

        xpx = (self.essencex.T * self.essencex + (self.essencex.T * self.essencex).T) / 2
        cpc = (self.c_matrix * self.c_matrix.T + (self.c_matrix * self.c_matrix.T).T) / 2
        upu = (self.u_matrix.T * self.u_matrix + (self.u_matrix.T * self.u_matrix).T) / 2
        rank_x = ranksymm(xpx, self.tolerance)  # R
        rank_c = ranksymm(cpc, self.tolerance)
        rank_u = ranksymm(upu, self.tolerance)
        min_rank_c_u = min(rank_c, rank_u)  # S
        xpxginv = np.linalg.pinv(xpx)  # (Moore-Penrose) pseudo-inverse, the same method in IML--GINV()
        m_matrix = self.c_matrix * xpxginv * self.c_matrix.T
        rank_m = ranksymm(m_matrix, self.tolerance)

        sigma_diag_inv = np.linalg.inv(np.diag(np.sqrt(np.diag(self.sigma))))
        rho = sigma_diag_inv * self.sigma * sigma_diag_inv

        # Check and warning for less than full rank ESSENCEX
        # R
        if rank_x != num_col_x:
            if self.opt_fracrepn:
                warnings.warn('WARNING 11: You are performing calculations using a less than full rank ESSENCEX '
                              'matrix.  In doing so, you accept responsibility for the results being the ones '
                              'desired.')
            else:
                raise Exception('ERROR 45: You have specified a less than full rank ESSENCEX matrix.  Specifying '
                                'opt_fracrepn=True will allow calculations to complete in this case.  In doing so, '
                                'you accept responsibility for the results being the ones desired.')

        # Check for testable GLH and estimaable CBETAU
        # Full rank C matrix
        if rank_c != num_row_c:
            raise Exception('ERROR 46: C must be full rank to ensure testability of CBETAU = THETA0.')

        # Full rank U matrix
        if rank_u != num_col_u:
            raise Exception('ERROR 47: U must be full rank to ensure testability of CBETAU = THETA0.')

        # C = C * XPXGINV * XPX
        if max(abs(self.c_matrix - self.c_matrix * xpxginv * xpx)) > self.tolerance:
            raise Exception('ERROR 48: A requirement for estimability is that C = C * GINV(X`X) * X`X.  Your choices '
                            'of C and ESSENCEX do not provide this.  We suggest using full rank coding for ESSENCEX.')

        if rank_m != num_row_c:
            raise Exception('ERROR 49: M = C(GINV(X`X))C` must be full rank to ensure testability of CBETAU = THETA0.')

        # Create Orthonormal U for UNIREP test power calculations
        u_orth = self.orthonormal_u()

        # TODO the whole opt stuff
        self.option_check(min_rank_c_u, num_col_u, num_row_c)

        #
        # prepare for power calculation
        # 1.sigma
        sigma_scaled = self.sigma * self.sigma_scalar
        variance_scaled = np.diag(sigma_scaled)
        if variance_scaled.min() <= self.tolerance:
            raise Exception('ERROR 52: The covariance matrix formed with SIGSCAL element has a variance <= '
                            'TOLERANCE (too small).')

        stdev_scaled = np.sqrt(variance_scaled)
        rho_scaled = np.diag(np.ones((np.shape(self.beta)[1], 1)) / stdev_scaled) * sigma_scaled \
                     * np.diag(np.ones((np.shape(self.beta)[1], 1)) / stdev_scaled)

        # 2.rhos
        rho_junk = rho_scaled * self.rho_scalar
        rho_offdiag = rho_junk - np.diag(np.diag(rho_junk))
        rho_offdiag_symm = (rho_offdiag + rho_offdiag.T) / 2

        if abs(rho_offdiag_symm).max() > 1:
            raise Exception('ERROR 53: SIGSCAL and RHOSCAL produce a correlation with absolute value > 1 .')
        if abs(rho_offdiag_symm).max() == 1:
            warnings.warn('WARNING 13: SIGSCAL and RHOSCAL produce a correlation with absolute value = 1 .')

        # create new sigma from rho
        sigma_star = u_orth.T * (np.diag(stdev_scaled) * (rho_offdiag_symm + np.identity(np.shape(self.beta)[1]))
                                 * np.diag(stdev_scaled)) * u_orth
        eigval_sigma_star, eigvec_sigma_star = np.linalg.eig(sigma_star)
        rank_sigma_star = ranksymm(sigma_star, self.tolerance)

        # 3.Beta
        beta_scaled = self.beta * self.beta_scalar
        theta = self.c_matrix * beta_scaled * u_orth
        essh = (theta - self.theta_zero).T * np.linalg.solve(m_matrix, np.identity(np.shape(m_matrix)[0])) \
               * (theta - self.theta_zero)

        # 4.N
        total_sample_size = self.rep_n * np.shape(self.essencex)[0]
        if np.int(total_sample_size) != total_sample_size:
            warnings.warn('WARNING 14: You have specified a design with non-integer N. In doing so, you are '
                          'responsible for correct interpretation of the results.')

        error_sum_square = sigma_star * (total_sample_size - rank_x)  # E
        hypo_sum_square = self.repn * essh  # H

        if num_col_u > total_sample_size - rank_x and \
                (total_sample_size - rank_x > 0 and
                     (self.opt_calc_collapse |
                          self.opt_calc_hlt |
                          self.opt_calc_pbt |
                          self.opt_calc_wlk |
                          self.opt_calc_un |
                          self.opt_calc_gg |
                          self.opt_calc_box)):
            raise Exception('ERROR 97: Tests GG, UN, Box, HLT, PBT, and WLK '
                            'have undesirable properties when applied to high '
                            'dimension low sample size data, thus are disallowed '
                            'when B > N-R. To turn off these tests, the user '
                            'should specify OPT_OFF={GG UN HLT PBT WLK}')
        if num_col_u <= total_sample_size - rank_x and rank_sigma_star != num_col_u:
            raise Exception('ERROR 54: SIGMASTAR = U`*SIGMA*U must be full rank to ensure testability of CBETAU = '
                            'THETA0.')

        # 5.Eigenvalues for H*INV(E)
        if self.opt_calc_collapse | self.opt_calc_hlt | self.opt_calc_pbt | self.opt_calc_wlk:
            # TODO can we raise an Exception for this case?
            if total_sample_size - rank_x <= 0:
                eval = float('nan')
                rhosq = float('nan')
                num_distinct_eigval = 1  # D
                mtp = num_col_u
                deigval = float('nan')
                slam1 = float('nan')
                slam2 = float('nan')
                slam3 = float('nan')
                cancorrmax = float('nan')
            else:
                inverse_error_sum = np.linalg.inv(np.linalg.cholesky(error_sum_square)).T
                hei_orth = inverse_error_sum * error_sum_square * inverse_error_sum.T
                hei_orth_symm = (hei_orth + hei_orth.T) / 2
                eval = np.linalg.eigvals(hei_orth)[0:min_rank_c_u]
                eval = eval * (eval > self.tolerance)

                # make vector of squared generalized canonical correlations
                eval_pop = eval * (total_sample_size - rank_x) / total_sample_size
                cancorr = np.round((eval_pop / (np.ones((1, min_rank_c_u)) + eval_pop)), self.round)
                cancorrmax = cancorr.max()

        #
        # Compute power
        # MultiRep
        if self.opt_calc_collapse:
            special_power = multirep.special(rank_C=rank_c,
                                             rank_U=rank_u,
                                             rank_X=rank_x,
                                             total_N=total_sample_size,
                                             eval_HINVE=eval,
                                             alphascalar=self.alpha,
                                             cl_type=self.CL.cl_type,
                                             n_est=self.CL.n_est,
                                             rank_est=self.CL.rank_est,
                                             alpha_cl=self.CL.alpha_cl,
                                             alpha_cu=self.CL.alpha_cu,
                                             tolerance=self.tolerance)
            return special_power

        if self.opt_calc_hlt:
            hlt_power = multirep.hlt(rank_C=rank_c,
                                     rank_U=rank_u,
                                     rank_X=rank_x,
                                     total_N=total_sample_size,
                                     eval_HINVE=eval,
                                     alphascalar=self.alpha,
                                     MultiHLT=self.MultiHLT,
                                     cl_type=self.CL.cl_type,
                                     n_est=self.CL.n_est,
                                     rank_est=self.CL.rank_est,
                                     alpha_cl=self.CL.alpha_cl,
                                     alpha_cu=self.CL.alpha_cu,
                                     tolerance=self.tolerance)
            return hlt_power

        if self.opt_calc_pbt:
            pbt_power = multirep.pbt(rank_C=rank_c,
                                     rank_U=rank_u,
                                     rank_X=rank_x,
                                     total_N=total_sample_size,
                                     eval_HINVE=eval,
                                     alphascalar=self.alpha,
                                     MultiPBT=self.MultiPBT,
                                     cl_type=self.CL.cl_type,
                                     n_est=self.CL.n_est,
                                     rank_est=self.CL.rank_est,
                                     alpha_cl=self.CL.alpha_cl,
                                     alpha_cu=self.CL.alpha_cu,
                                     tolerance=self.tolerance)
            return pbt_power

        if self.opt_calc_wlk:
            wlk_power = multirep.wlk(rank_C=rank_c,
                                     rank_U=rank_u,
                                     rank_X=rank_x,
                                     total_N=total_sample_size,
                                     eval_HINVE=eval,
                                     alphascalar=self.alpha,
                                     MultiWLK=self.MultiWLK,
                                     cl_type=self.CL.cl_type,
                                     n_est=self.CL.n_est,
                                     rank_est=self.CL.rank_est,
                                     alpha_cl=self.CL.alpha_cl,
                                     alpha_cu=self.CL.alpha_cu,
                                     tolerance=self.tolerance)
            return wlk_power

        # UniRep
        if self.opt_calc_un | self.opt_calc_hf | self.opt_calc_cm | self.opt_calc_gg | self.opt_calc_box:
            d, mtp, eps, deigval, slam1, slam2, slam3 = unirep.firstuni(sigma_star, num_col_u)

        if self.opt_calc_un:
            un_power = unirep.lastuni(rank_C=rank_c,
                                      rank_U=rank_u,
                                      total_N=total_sample_size,
                                      rank_X=rank_x,
                                      error_sum_square=error_sum_square,
                                      hypo_sum_square=hypo_sum_square,
                                      sig_type=self.CL.sigma_type,
                                      ip_plan=self.IP.ip_plan,
                                      rank_ip=self.IP.rank_ip,
                                      n_est=self.CL.n_est,
                                      rank_est=self.CL.rank_est,
                                      n_ip=self.IP.n_ip,
                                      sigmastareval=eigval_sigma_star,
                                      sigmastarevec=eigvec_sigma_star,
                                      cl_type=self.CL.cl_type,
                                      alpha_cl=self.CL.alpha_cl,
                                      alpha_cu=self.CL.alpha_cu,
                                      tolerance=self.tolerance,
                                      round=self.round,
                                      exeps=1,
                                      eps=eps,
                                      alpha_scalar=self.alpha,
                                      opt_calc_un=self.opt_calc_un,
                                      opt_calc_gg=self.opt_calc_gg,
                                      opt_calc_box=self.opt_calc_box,
                                      opt_calc_hf=self.opt_calc_hf,
                                      opt_calc_cm=self.opt_calc_cm,
                                      unirepmethod=self.UnirepUncorrected)
            return un_power

        if total_sample_size - rank_x <= 0:
            exeps = float('nan')
        else:
            if self.opt_calc_hf:
                exeps = unirep.hfexeps(sigma_star, rank_u, total_sample_size, rank_x, self.UnirepUncorrected)
                if self.IP.ip_plan:
                    eps = unirep.hfexeps(sigma_star, rank_u, self.IP.n_ip, self.IP.rank_ip, self.UnirepUncorrected)
                hf_power = unirep.lastuni(rank_C=rank_c,
                                          rank_U=rank_u,
                                          total_N=total_sample_size,
                                          rank_X=rank_x,
                                          error_sum_square=error_sum_square,
                                          hypo_sum_square=hypo_sum_square,
                                          sig_type=self.CL.sigma_type,
                                          ip_plan=self.IP.ip_plan,
                                          rank_ip=self.IP.rank_ip,
                                          n_est=self.CL.n_est,
                                          rank_est=self.CL.rank_est,
                                          n_ip=self.IP.n_ip,
                                          sigmastareval=eigval_sigma_star,
                                          sigmastarevec=eigvec_sigma_star,
                                          cl_type=self.CL.cl_type,
                                          alpha_cl=self.CL.alpha_cl,
                                          alpha_cu=self.CL.alpha_cu,
                                          tolerance=self.tolerance,
                                          round=self.round,
                                          exeps=exeps,
                                          eps=eps,
                                          alpha_scalar=self.alpha,
                                          opt_calc_un=self.opt_calc_un,
                                          opt_calc_gg=self.opt_calc_gg,
                                          opt_calc_box=self.opt_calc_box,
                                          opt_calc_hf=self.opt_calc_hf,
                                          opt_calc_cm=self.opt_calc_cm,
                                          unirepmethod=self.UnirepHuynhFeldt)
                return hf_power

            if self.opt_calc_cm:
                exeps = unirep.cmexeps(sigma_star, rank_u, total_sample_size, rank_x, self.UnirepUncorrected)
                if self.IP.ip_plan:
                    eps = unirep.cmexeps(sigma_star, rank_u, self.IP.n_ip, self.IP.rank_ip, self.UnirepUncorrected)
                cm_power = unirep.lastuni(rank_C=rank_c,
                                          rank_U=rank_u,
                                          total_N=total_sample_size,
                                          rank_X=rank_x,
                                          error_sum_square=error_sum_square,
                                          hypo_sum_square=hypo_sum_square,
                                          sig_type=self.CL.sigma_type,
                                          ip_plan=self.IP.ip_plan,
                                          rank_ip=self.IP.rank_ip,
                                          n_est=self.CL.n_est,
                                          rank_est=self.CL.rank_est,
                                          n_ip=self.IP.n_ip,
                                          sigmastareval=eigval_sigma_star,
                                          sigmastarevec=eigvec_sigma_star,
                                          cl_type=self.CL.cl_type,
                                          alpha_cl=self.CL.alpha_cl,
                                          alpha_cu=self.CL.alpha_cu,
                                          tolerance=self.tolerance,
                                          round=self.round,
                                          exeps=exeps,
                                          eps=eps,
                                          alpha_scalar=self.alpha,
                                          opt_calc_un=self.opt_calc_un,
                                          opt_calc_gg=self.opt_calc_gg,
                                          opt_calc_box=self.opt_calc_box,
                                          opt_calc_hf=self.opt_calc_hf,
                                          opt_calc_cm=self.opt_calc_cm,
                                          unirepmethod=self.UnirepHuynhFeldtChiMuller)
                return cm_power

            if self.opt_calc_gg:
                exeps = unirep.ggexeps(sigma_star, rank_u, total_sample_size, rank_x, self.UnirepHuynhFeldt)
                if self.IP.ip_plan:
                    eps = unirep.ggexeps(sigma_star, rank_u, self.IP.n_ip, self.IP.rank_ip, self.UnirepHuynhFeldt)
                gg_power = unirep.lastuni(rank_C=rank_c,
                                          rank_U=rank_u,
                                          total_N=total_sample_size,
                                          rank_X=rank_x,
                                          error_sum_square=error_sum_square,
                                          hypo_sum_square=hypo_sum_square,
                                          sig_type=self.CL.sigma_type,
                                          ip_plan=self.IP.ip_plan,
                                          rank_ip=self.IP.rank_ip,
                                          n_est=self.CL.n_est,
                                          rank_est=self.CL.rank_est,
                                          n_ip=self.IP.n_ip,
                                          sigmastareval=eigval_sigma_star,
                                          sigmastarevec=eigvec_sigma_star,
                                          cl_type=self.CL.cl_type,
                                          alpha_cl=self.CL.alpha_cl,
                                          alpha_cu=self.CL.alpha_cu,
                                          tolerance=self.tolerance,
                                          round=self.round,
                                          exeps=exeps,
                                          eps=eps,
                                          alpha_scalar=self.alpha,
                                          opt_calc_un=self.opt_calc_un,
                                          opt_calc_gg=self.opt_calc_gg,
                                          opt_calc_box=self.opt_calc_box,
                                          opt_calc_hf=self.opt_calc_hf,
                                          opt_calc_cm=self.opt_calc_cm,
                                          unirepmethod=self.UnirepGeisserGreenhouse)
                return gg_power

            if self.opt_calc_box:
                exeps = 1 / num_col_u
                box_power = unirep.lastuni(rank_C=rank_c,
                                           rank_U=rank_u,
                                           total_N=total_sample_size,
                                           rank_X=rank_x,
                                           error_sum_square=error_sum_square,
                                           hypo_sum_square=hypo_sum_square,
                                           sig_type=self.CL.sigma_type,
                                           ip_plan=self.IP.ip_plan,
                                           rank_ip=self.IP.rank_ip,
                                           n_est=self.CL.n_est,
                                           rank_est=self.CL.rank_est,
                                           n_ip=self.IP.n_ip,
                                           sigmastareval=eigval_sigma_star,
                                           sigmastarevec=eigvec_sigma_star,
                                           cl_type=self.CL.cl_type,
                                           alpha_cl=self.CL.alpha_cl,
                                           alpha_cu=self.CL.alpha_cu,
                                           tolerance=self.tolerance,
                                           round=self.round,
                                           exeps=exeps,
                                           eps=eps,
                                           alpha_scalar=self.alpha,
                                           opt_calc_un=self.opt_calc_un,
                                           opt_calc_gg=self.opt_calc_gg,
                                           opt_calc_box=self.opt_calc_box,
                                           opt_calc_hf=self.opt_calc_hf,
                                           opt_calc_cm=self.opt_calc_cm,
                                           unirepmethod=self.UnirepBox)
                return box_power

    def orthonormal_u(self):
        if self.opt_uniforce:
            warnings.warn('WARNING 17: You have specified the option UNIFORCE, which allows power calculations to '
                          'continue without a U matrix that is orthonormal and orthogonal to a Px1 column of 1. This '
                          'option should be used WITH CAUTION. The user accepts responsibility for the results being '
                          'the ones desired.')
        u_orth = self.u_matrix
        if self.opt_calc_un or \
                self.opt_calc_hf or \
                self.opt_calc_cm or \
                self.opt_calc_gg or \
                self.opt_calc_box:

            upu = (self.u_matrix.T * self.u_matrix + (self.u_matrix.T * self.u_matrix).T) / 2

            if upu[0, 0] != 0:
                upu = upu / upu[0, 0]
            udif = abs(upu - np.identity(np.shape(self.u_matrix)[1]))

            if (max(udif) > np.sqrt(self.tolerance)) or \
                    (np.shape(self.u_matrix)[1] > 1 and max(
                            self.u_matrix.T * np.ones((np.shape(self.beta)[1], 1))) > np.sqrt(self.tolerance)):
                if not self.opt_orthu and not self.opt_uniforce:
                    raise Exception('ERROR 50: For univariate repeated measures, U must be proportional to an '
                                    'orthonormal matrix [U`U = cI] and orthogonal to a Px1 column of 1 [U`1 = 0]. The '
                                    'U matrix specified does not have these properties. To have this program provide '
                                    'a U matrix with the required properties, specify OPT_ON= {ORTHU}; . To allow '
                                    'power calculations to continue without a U matrix with these properties, '
                                    'specify OPT_ON= {UNIFORCE};  If you do not wish to compute power for UNIREP '
                                    'tests, specify OPT_OFF = {GG HF UN BOX}; .')
                if self.opt_orthu and not self.opt_uniforce:
                    # TODO QR decomposition, Gram-Schmidt orthonormal factorization difference
                    u_orth, t_matrix = np.linalg.qr(self.u_matrix)
                    if (np.shape(self.u_matrix)[1] > 1 and max(
                                u_orth.T * np.ones((np.shape(self.beta)[1], 1))) > np.sqrt(self.tolerance)):
                        raise Exception('ERROR 51: You have specified option ORTHU so that the program will provide a '
                                        'U that is proportional to to an orthonormal matrix and orthogonal to a Px1 '
                                        'column of 1. The original U given cannot be made to be orthogonal to a Px1 '
                                        'column of 1. Choose a different U matrix.')
                    cbetau = self.c_matrix * self.beta * u_orth
                    warnings.warn('WARNING 12: For univariate repeated measures, U must be proportional to an '
                                  'orthonormal matrix [U`U = cI] and orthogonal to a Px1 column of 1 [U`1 = 0]. The U '
                                  'matrix specified does not have these properties.  A new U matrix with these '
                                  'properties was created from your input and used in UNIREP calculations')

        return u_orth

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

        # Create B for HDLSS power calculations with the CM option
        if self.opt_cmwarn and num_col_u > 3000:
            raise Exception('ERROR 96: Current common computer memory size allows computing power for HDLSS models '
                            'with <= 3000 repeated measures. Computing power for models with B>3000 may lead to '
                            'program failure due to insufficient memory. Models with B>3000 may be run by specifying '
                            'option opt_cmwarn=False; however, turning off this option should be done WITH CAUTION. '
                            'The user accepts responsbility for potential program failure due to insufficient memory.')

        if not self.opt_cmwarn and num_col_u > 3000:
            warnings.warn('WARNING 18: You have turned off option CMWARN, allowing power to be computed for HDLSS '
                          'models with >3000 repeated measures. Computing power for models with B>3000 may lead to '
                          'program failure due to insufficient memory, thus turning off this option should be done '
                          'WITH CAUTION. The user accepts responsbility for potential program failure due to '
                          'insufficient memory.')
