import numpy as np
import warnings

from glmpowercalc import unirep, multirep
from glmpowercalc.constants import Constants
from glmpowercalc.ranksymm import ranksymm


class PowerReturn:
    def __init__(self, special_power, hlt_power, pbt_power, wlk_power, un_power, hf_power, cm_power, gg_power,
                 box_power):
        self.special_power = special_power
        self.hlt_power = hlt_power
        self.pbt_power = pbt_power
        self.wlk_power = wlk_power
        self.un_power = un_power
        self.hf_power = hf_power
        self.cm_power = cm_power
        self.gg_power = gg_power
        self.box_power = box_power


def power(essencex, beta, c_matrix, u_matrix, sigma, theta_zero, Scalar, CalcMethod, Option, CL, IP):
    if CL.cl_type == Constants.CLTYPE_NOT_DESIRED and Option.opt_noncencl:
        raise Exception("ERROR 83: NONCENCL is not a valid option when CL not desired.")

    # Check repn
    if Scalar.rep_n <= Scalar.tolerance:
        raise Exception('ERROR 10: All REPN values must be > TOLERANCE > 0.')

    if Option.opt_fracrepn and Scalar.rep_n % 1 != 1:
        raise Exception('ERROR 11: All REPN values must be positive integers. To allow fractional REPN values, '
                        'specify opt_fracrepn')

    # Check sigscal
    if Scalar.sigma_scalar <= Scalar.tolerance:
        raise Exception('ERROR 12: All SIGSCAL values must be > TOLERANCE > 0.')

    # Check alpha
    if Scalar.alpha <= Scalar.tolerance or Scalar.alpha >= 1:
        raise Exception('ERROR 13: All ALPHA values must be > TOLERANCE > 0 and < 1.')

    # Check tolerance
    if Scalar.tolerance <= 0:
        raise Exception('ERROR 17: User specified TOLERANCE <= zero.')
    if Scalar.tolerance >= 0.01:
        raise Exception('WARNING 6: User specified TOLERANCE >= 0.01. This is the value assumed to be numeric '
                        'zero and affects many calculations. Please check that this value is correct.')

    # Check UCDF
    if CalcMethod.UnirepUncorrected == Constants.UCDF_MULLER1989_APPROXIMATION or \
                    CalcMethod.UnirepHuynhFeldt == Constants.UCDF_MULLER1989_APPROXIMATION or \
                    CalcMethod.UnirepHuynhFeldtChiMuller == Constants.UCDF_MULLER1989_APPROXIMATION or \
                    CalcMethod.UnirepGeisserGreenhouse == Constants.UCDF_MULLER1989_APPROXIMATION or \
                    CalcMethod.UnirepBox == Constants.UCDF_MULLER1989_APPROXIMATION:
        warnings.warn('WARNING 7: You have chosen the Muller, Barton (1989) approximation for the UNIREP '
                      'statistic CDF. Muller, Edwards, Taylor (2004) found NO condition where their approximation '
                      'was not superior to this Muller, Barton approximation.  Suggest specifying '
                      'UCDF_MULLER2004_APPROXIMATION; '
                      'unless you are performing a backwards comparison calculation.')

    # Check IP_PLAN and SIGTYPE
    if IP.ip_plan and CL.sigma_type:
        raise Exception('ERROR 91: SIGMA must be known when planning an internal pilot.')

    # define initial parameters for power calculation
    num_col_x = np.shape(essencex)[1]  # Q
    num_row_c = np.shape(c_matrix)[0]  # A
    num_col_u = np.shape(u_matrix)[1]  # B
    num_response = np.shape(beta)[1]  # P

    # Check on size and conformity of matrices
    if np.diag(sigma).min() < -Scalar.tolerance:
        raise Exception('At least one variance < -tolerance. Check SIGMA')
    if np.shape(u_matrix)[0] != np.shape(sigma)[0]:
        raise Exception('ERROR 37: The number of rows of U is not equal to number of rows of SIGMA.')
    if np.shape(c_matrix)[1] != np.shape(beta)[0]:
        raise Exception('ERROR 38: The number of columns of C is not equal to number of rows of BETA.')
    if np.shape(beta)[1] != np.shape(u_matrix)[0]:
        raise Exception('ERROR 39: The number of columns of BETA is not equal to number of rows of U.')

    # Q
    if num_col_x != np.shape(beta)[0]:
        raise Exception('ERROR 40: The number of columns of ESSENCEX is not equal to number of rows of BETA.')

    # A
    if num_row_c > np.shape(c_matrix)[1]:
        raise Exception('ERROR 41: The number of rows of C is greater than the number of columns of C.')

    # B
    if num_col_u > np.shape(u_matrix)[0]:
        raise Exception('ERROR 42: The number of columns of U is greater than the number of rows of U.')

    if np.shape(theta_zero)[0] != num_row_c or \
                    np.shape(theta_zero)[1] != num_col_u:
        raise Exception('ERROR 43: The THETA0 matrix does not conform to CBETAU.')

    if Scalar.sigma_scalar <= Scalar.tolerance:
        raise Exception('ERROR 44: Smallest value in SIGSCAL <= TOLERANCE (too small)')

    xpx = (essencex.T * essencex + (essencex.T * essencex).T) / 2
    cpc = (c_matrix * c_matrix.T + (c_matrix * c_matrix.T).T) / 2
    upu = (u_matrix.T * u_matrix + (u_matrix.T * u_matrix).T) / 2
    rank_x = ranksymm(xpx, Scalar.tolerance)  # R
    rank_c = ranksymm(cpc, Scalar.tolerance)
    rank_u = ranksymm(upu, Scalar.tolerance)
    min_rank_c_u = min(rank_c, rank_u)  # S
    xpxginv = np.linalg.pinv(xpx)  # (Moore-Penrose) pseudo-inverse, the same method in IML--GINV()
    m_matrix = c_matrix * xpxginv * c_matrix.T
    rank_m = ranksymm(m_matrix, Scalar.tolerance)

    sigma_diag_inv = np.linalg.inv(np.diag(np.sqrt(np.diag(sigma))))
    rho = sigma_diag_inv * sigma * sigma_diag_inv

    # Check and warning for less than full rank ESSENCEX
    # R
    if rank_x != num_col_x:
        if Option.opt_fracrepn:
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
    if max(abs(c_matrix - c_matrix * xpxginv * xpx)) > Scalar.tolerance:
        raise Exception('ERROR 48: A requirement for estimability is that C = C * GINV(X`X) * X`X.  Your choices '
                        'of C and ESSENCEX do not provide this.  We suggest using full rank coding for ESSENCEX.')

    if rank_m != num_row_c:
        raise Exception('ERROR 49: M = C(GINV(X`X))C` must be full rank to ensure testability of CBETAU = THETA0.')

    # Create Orthonormal U for UNIREP test power calculations
    if Option.opt_uniforce:
        warnings.warn('WARNING 17: You have specified the option UNIFORCE, which allows power calculations to '
                      'continue without a U matrix that is orthonormal and orthogonal to a Px1 column of 1. This '
                      'option should be used WITH CAUTION. The user accepts responsibility for the results being '
                      'the ones desired.')
    u_orth = u_matrix
    if Option.opt_calc_un or \
            Option.opt_calc_hf or \
            Option.opt_calc_cm or \
            Option.opt_calc_gg or \
            Option.opt_calc_box:

        upu = (u_matrix.T * u_matrix + (u_matrix.T * u_matrix).T) / 2

        if upu[0, 0] != 0:
            upu = upu / upu[0, 0]
        udif = abs(upu - np.identity(np.shape(u_matrix)[1]))

        if (max(udif) > np.sqrt(Scalar.tolerance)) or \
                (np.shape(u_matrix)[1] > 1 and max(
                        u_matrix.T * np.ones((np.shape(beta)[1], 1))) > np.sqrt(Scalar.tolerance)):
            if not Option.opt_orthu and not Option.opt_uniforce:
                raise Exception('ERROR 50: For univariate repeated measures, U must be proportional to an '
                                'orthonormal matrix [U`U = cI] and orthogonal to a Px1 column of 1 [U`1 = 0]. The '
                                'U matrix specified does not have these properties. To have this program provide '
                                'a U matrix with the required properties, specify OPT_ON= {ORTHU}; . To allow '
                                'power calculations to continue without a U matrix with these properties, '
                                'specify OPT_ON= {UNIFORCE};  If you do not wish to compute power for UNIREP '
                                'tests, specify OPT_OFF = {GG HF UN BOX}; .')
            if Option.opt_orthu and not Option.opt_uniforce:
                # TODO QR decomposition, Gram-Schmidt orthonormal factorization difference
                u_orth, t_matrix = np.linalg.qr(u_matrix)
                if (np.shape(u_matrix)[1] > 1 and max(
                            u_orth.T * np.ones((np.shape(beta)[1], 1))) > np.sqrt(Scalar.tolerance)):
                    raise Exception('ERROR 51: You have specified option ORTHU so that the program will provide a '
                                    'U that is proportional to to an orthonormal matrix and orthogonal to a Px1 '
                                    'column of 1. The original U given cannot be made to be orthogonal to a Px1 '
                                    'column of 1. Choose a different U matrix.')
                cbetau = c_matrix * beta * u_orth
                warnings.warn('WARNING 12: For univariate repeated measures, U must be proportional to an '
                              'orthonormal matrix [U`U = cI] and orthogonal to a Px1 column of 1 [U`1 = 0]. The U '
                              'matrix specified does not have these properties.  A new U matrix with these '
                              'properties was created from your input and used in UNIREP calculations')

    # TODO the whole opt stuff
    if not (Option.opt_calc_collapse |
                Option.opt_calc_hlt |
                Option.opt_calc_pbt |
                Option.opt_calc_wlk |
                Option.opt_calc_un |
                Option.opt_calc_hf |
                Option.opt_calc_cm |
                Option.opt_calc_gg |
                Option.opt_calc_box):
        raise Exception("ERROR 9: No power calculation selected.")
    if Option.opt_calc_collapse:
        if min_rank_c_u > 1:
            Option.opt_calc_collapse = False
            warnings.warn("WARNING 1: Rank(C*BETA*U) > 1, so COLLAPSE option ignored.")
        if num_col_u == 1:
            Option.opt_calc_hlt = 0
            Option.opt_calc_pbt = 0
            Option.opt_calc_wlk = 0
            Option.opt_calc_un = 0
            Option.opt_calc_hf = 0
            Option.opt_calc_cm = 0
            Option.opt_calc_gg = 0
            Option.opt_calc_box = 0
            warnings.warn(
                "WARNING 2: B = 1, so that all tests coincide (univariate case). "
                "Since collapse option is on, power is given as one value with the heading POWER. "
                "To print powers with a heading for  each test, specify collapse option off.")
        if num_col_u > 1 and num_row_c == 1:
            if Option.opt_calc_hlt == 0 and \
                            Option.opt_calc_pbt == 0 and \
                            Option.opt_calc_wlk == 0:
                pass

    # Create B for HDLSS power calculations with the CM option
    if Option.opt_cmwarn and num_col_u > 3000:
        raise Exception('ERROR 96: Current common computer memory size allows computing power for HDLSS models '
                        'with <= 3000 repeated measures. Computing power for models with B>3000 may lead to '
                        'program failure due to insufficient memory. Models with B>3000 may be run by specifying '
                        'option opt_cmwarn=False; however, turning off this option should be done WITH CAUTION. '
                        'The user accepts responsbility for potential program failure due to insufficient memory.')

    if not Option.opt_cmwarn and num_col_u > 3000:
        warnings.warn('WARNING 18: You have turned off option CMWARN, allowing power to be computed for HDLSS '
                      'models with >3000 repeated measures. Computing power for models with B>3000 may lead to '
                      'program failure due to insufficient memory, thus turning off this option should be done '
                      'WITH CAUTION. The user accepts responsbility for potential program failure due to '
                      'insufficient memory.')

    #
    # prepare for power calculation
    # 1.sigma
    sigma_scaled = sigma * Scalar.sigma_scalar
    variance_scaled = np.diag(sigma_scaled)
    if variance_scaled.min() <= Scalar.tolerance:
        raise Exception('ERROR 52: The covariance matrix formed with SIGSCAL element has a variance <= '
                        'TOLERANCE (too small).')

    stdev_scaled = np.sqrt(variance_scaled)
    rho_scaled = np.diag(np.ones((np.shape(beta)[1], 1)) / stdev_scaled) * sigma_scaled * np.diag(
        np.ones((np.shape(beta)[1], 1)) / stdev_scaled)

    # 2.rhos
    rho_junk = rho_scaled * Scalar.rho_scalar
    rho_offdiag = rho_junk - np.diag(np.diag(rho_junk))
    rho_offdiag_symm = (rho_offdiag + rho_offdiag.T) / 2

    if abs(rho_offdiag_symm).max() > 1:
        raise Exception('ERROR 53: SIGSCAL and RHOSCAL produce a correlation with absolute value > 1 .')
    if abs(rho_offdiag_symm).max() == 1:
        warnings.warn('WARNING 13: SIGSCAL and RHOSCAL produce a correlation with absolute value = 1 .')

    # create new sigma from rho
    sigma_star = u_orth.T * (np.diag(stdev_scaled) * (rho_offdiag_symm + np.identity(np.shape(beta)[1]))
                             * np.diag(stdev_scaled)) * u_orth
    eigval_sigma_star, eigvec_sigma_star = np.linalg.eig(sigma_star)
    rank_sigma_star = ranksymm(sigma_star, Scalar.tolerance)

    # 3.Beta
    beta_scaled = beta * Scalar.beta_scalar
    theta = c_matrix * beta_scaled * u_orth
    essh = (theta - theta_zero).T * np.linalg.solve(m_matrix, np.identity(np.shape(m_matrix)[0])) \
           * (theta - theta_zero)

    # 4.N
    total_sample_size = Scalar.rep_n * np.shape(essencex)[0]
    if np.int(total_sample_size) != total_sample_size:
        warnings.warn('WARNING 14: You have specified a design with non-integer N. In doing so, you are '
                      'responsible for correct interpretation of the results.')

    error_sum_square = sigma_star * (total_sample_size - rank_x)  # E
    hypo_sum_square = Scalar.rep_n * essh  # H

    if num_col_u > total_sample_size - rank_x and \
            (total_sample_size - rank_x > 0 and
                 (Option.opt_calc_collapse |
                      Option.opt_calc_hlt |
                      Option.opt_calc_pbt |
                      Option.opt_calc_wlk |
                      Option.opt_calc_un |
                      Option.opt_calc_gg |
                      Option.opt_calc_box)):
        raise Exception('ERROR 97: Tests GG, UN, Box, HLT, PBT, and WLK '
                        'have undesirable properties when applied to high '
                        'dimension low sample size data, thus are disallowed '
                        'when B > N-R. To turn off these tests, the user '
                        'should specify OPT_OFF={GG UN HLT PBT WLK}')
    if total_sample_size - rank_x >= num_col_u != rank_sigma_star:
        raise Exception('ERROR 54: SIGMASTAR = U`*SIGMA*U must be full rank to ensure testability of CBETAU = '
                        'THETA0.')

    # 5.Eigenvalues for H*INV(E)
    if Option.opt_calc_collapse | Option.opt_calc_hlt | Option.opt_calc_pbt | Option.opt_calc_wlk:
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
            inverse_error_sum = np.linalg.inv(np.linalg.cholesky(error_sum_square))
            hei_orth = inverse_error_sum * hypo_sum_square * inverse_error_sum.T
            hei_orth_symm = (hei_orth + hei_orth.T) / 2
            eval = np.linalg.eigvals(hei_orth_symm)[0:min_rank_c_u]
            eval = eval * (eval > Scalar.tolerance)

            # make vector of squared generalized canonical correlations
            eval_pop = eval * (total_sample_size - rank_x) / total_sample_size
            cancorr = (eval_pop / (np.ones((1, min_rank_c_u)) + eval_pop))
            cancorrmax = cancorr.max()

    #
    # Compute power
    # MultiRep
    if Option.opt_calc_collapse:
        special_power = multirep.special(rank_C=rank_c,
                                         rank_U=rank_u,
                                         rank_X=rank_x,
                                         total_N=total_sample_size,
                                         eval_HINVE=eval,
                                         CL=CL,
                                         Scalar=Scalar)
    else:
        special_power = None

    if Option.opt_calc_hlt:
        hlt_power = multirep.hlt(rank_C=rank_c,
                                 rank_U=rank_u,
                                 rank_X=rank_x,
                                 total_N=total_sample_size,
                                 eval_HINVE=eval,
                                 MultiHLT=CalcMethod.MultiHLT,
                                 CL=CL,
                                 Scalar=Scalar)
    else:
        hlt_power = None

    if Option.opt_calc_pbt:
        pbt_power = multirep.pbt(rank_C=rank_c,
                                 rank_U=rank_u,
                                 rank_X=rank_x,
                                 total_N=total_sample_size,
                                 eval_HINVE=eval,
                                 MultiPBT=CalcMethod.MultiPBT,
                                 CL=CL,
                                 Scalar=Scalar)
    else:
        pbt_power = None

    if Option.opt_calc_wlk:
        wlk_power = multirep.wlk(rank_C=rank_c,
                                 rank_U=rank_u,
                                 rank_X=rank_x,
                                 total_N=total_sample_size,
                                 eval_HINVE=eval,
                                 MultiWLK=CalcMethod.MultiWLK,
                                 CL=CL,
                                 Scalar=Scalar)
    else:
        wlk_power = None

    # UniRep
    if Option.opt_calc_un | Option.opt_calc_hf | Option.opt_calc_cm | Option.opt_calc_gg | Option.opt_calc_box:
        d, mtp, eps, deigval, slam1, slam2, slam3 = unirep.firstuni(sigma_star, num_col_u)

    if Option.opt_calc_un:
        un_power = unirep.lastuni(rank_C=rank_c,
                                  rank_U=rank_u,
                                  total_N=total_sample_size,
                                  rank_X=rank_x,
                                  error_sum_square=error_sum_square,
                                  hypo_sum_square=hypo_sum_square,
                                  sigmastareval=eigval_sigma_star,
                                  sigmastarevec=eigvec_sigma_star,
                                  exeps=1,
                                  eps=eps,
                                  unirepmethod=CalcMethod.UnirepUncorrected,
                                  Scalar=Scalar,
                                  Option=Option,
                                  CL=CL,
                                  IP=IP)
    else:
        un_power = None

    if total_sample_size - rank_x <= 0:
        raise Exception("total sample size <= rank_x")

    if Option.opt_calc_hf:
        exeps = unirep.hfexeps(sigma_star, rank_u, total_sample_size, rank_x, CalcMethod.UnirepUncorrected)
        if IP.ip_plan:
            eps = unirep.hfexeps(sigma_star, rank_u, IP.n_ip, IP.rank_ip,
                                 CalcMethod.UnirepUncorrected)
        hf_power = unirep.lastuni(rank_C=rank_c,
                                  rank_U=rank_u,
                                  total_N=total_sample_size,
                                  rank_X=rank_x,
                                  error_sum_square=error_sum_square,
                                  hypo_sum_square=hypo_sum_square,
                                  sigmastareval=eigval_sigma_star,
                                  sigmastarevec=eigvec_sigma_star,
                                  exeps=exeps,
                                  eps=eps,
                                  unirepmethod=CalcMethod.UnirepHuynhFeldt,
                                  Scalar=Scalar,
                                  Option=Option,
                                  CL=CL,
                                  IP=IP)
    else:
        hf_power = None

    if Option.opt_calc_cm:
        exeps = unirep.cmexeps(sigma_star, rank_u, total_sample_size, rank_x, CalcMethod.UnirepUncorrected)
        if IP.ip_plan:
            eps = unirep.cmexeps(sigma_star, rank_u, IP.n_ip, IP.rank_ip,
                                 CalcMethod.UnirepUncorrected)
        cm_power = unirep.lastuni(rank_C=rank_c,
                                  rank_U=rank_u,
                                  total_N=total_sample_size,
                                  rank_X=rank_x,
                                  error_sum_square=error_sum_square,
                                  hypo_sum_square=hypo_sum_square,
                                  sigmastareval=eigval_sigma_star,
                                  sigmastarevec=eigvec_sigma_star,
                                  exeps=exeps,
                                  eps=eps,
                                  unirepmethod=CalcMethod.UnirepHuynhFeldtChiMuller,
                                  Scalar=Scalar,
                                  Option=Option,
                                  CL=CL,
                                  IP=IP)
    else:
        cm_power = None

    if Option.opt_calc_gg:
        exeps = unirep.ggexeps(sigma_star, rank_u, total_sample_size, rank_x, CalcMethod.UnirepHuynhFeldt)
        if IP.ip_plan:
            eps = unirep.ggexeps(sigma_star, rank_u, IP.n_ip, IP.rank_ip,
                                 CalcMethod.UnirepHuynhFeldt)
        gg_power = unirep.lastuni(rank_C=rank_c,
                                  rank_U=rank_u,
                                  total_N=total_sample_size,
                                  rank_X=rank_x,
                                  error_sum_square=error_sum_square,
                                  hypo_sum_square=hypo_sum_square,
                                  sigmastareval=eigval_sigma_star,
                                  sigmastarevec=eigvec_sigma_star,
                                  exeps=exeps,
                                  eps=eps,
                                  unirepmethod=CalcMethod.UnirepGeisserGreenhouse,
                                  Scalar=Scalar,
                                  Option=Option,
                                  CL=CL,
                                  IP=IP)
    else:
        gg_power = None

    if Option.opt_calc_box:
        exeps = 1 / num_col_u
        box_power = unirep.lastuni(rank_C=rank_c,
                                   rank_U=rank_u,
                                   total_N=total_sample_size,
                                   rank_X=rank_x,
                                   error_sum_square=error_sum_square,
                                   hypo_sum_square=hypo_sum_square,
                                   sigmastareval=eigval_sigma_star,
                                   sigmastarevec=eigvec_sigma_star,
                                   exeps=exeps,
                                   eps=eps,
                                   unirepmethod=CalcMethod.UnirepBox,
                                   Scalar=Scalar,
                                   Option=Option,
                                   CL=CL,
                                   IP=IP)
    else:
        box_power = None

    return PowerReturn(special_power, hlt_power, pbt_power, wlk_power, un_power, hf_power, cm_power, gg_power,
                       box_power)
