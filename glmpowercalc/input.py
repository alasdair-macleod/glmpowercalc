from glmpowercalc.constants import Constants

class Scalar:
    def __init__(self, alpha=0.05, rep_n=1, beta_scalar=1, rho_scalar=1, sigma_scalar=1, tolerance=1e-12):
        self.alpha = alpha
        self.rep_n = rep_n
        self.beta_scalar = beta_scalar
        self.rho_scalar = rho_scalar
        self.sigma_scalar = sigma_scalar
        self.tolerance = tolerance

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


class CalcMethod:
    def __init__(self, unirepuncorrected=Constants.UCDF_MULLER2004_APPROXIMATION,
                 unirephuynhfeldt=Constants.UCDF_MULLER2004_APPROXIMATION,
                 unirephuynhfeldtchimuller=Constants.UCDF_MULLER2004_APPROXIMATION,
                 unirepgeissergreenhouse=Constants.UCDF_MULLER2004_APPROXIMATION,
                 unirepbox=Constants.UCDF_MULLER2004_APPROXIMATION,
                 epsilonapphuynhfeldt=Constants.EPSILON_MULLER2004,
                 epsilonapphuynhfeldtchimuller=Constants.EPSILON_MULLER2004,
                 epsilonappgeissergreenhouse=Constants.EPSILON_MULLER2004,
                 multihlt=Constants.MULTI_HLT_MCKEON_OS,
                 multipbt=Constants.MULTI_PBT_MULLER,
                 multiwlk=Constants.MULTI_WLK_RAO):
        self.UnirepUncorrected = unirepuncorrected
        self.UnirepHuynhFeldt = unirephuynhfeldt
        self.UnirepHuynhFeldtChiMuller = unirephuynhfeldtchimuller
        self.UnirepGeisserGreenhouse = unirepgeissergreenhouse
        self.UnirepBox = unirepbox
        self.EpsilonAppHuynhFeldt = epsilonapphuynhfeldt
        self.EpsilonAppHuynhFeldtChiMuller = epsilonapphuynhfeldtchimuller
        self.EpsilonAppGeisserGreenhouse = epsilonappgeissergreenhouse
        self.MultiHLT = multihlt
        self.MultiPBT = multipbt
        self.MultiWLK = multiwlk


class Option:
    def __init__(self, opt_noncencl=False,
                 opt_calc_collapse=False,
                 opt_calc_hlt=True,
                 opt_calc_pbt=True,
                 opt_calc_wlk=True,
                 opt_calc_un=False,
                 opt_calc_hf=False,
                 opt_calc_cm=False,
                 opt_calc_gg=True,
                 opt_calc_box=False,
                 opt_fracrepn=False,
                 opt_orthu=True,
                 opt_uniforce=False,
                 opt_cmwarn=True):
        self.opt_noncencl = opt_noncencl
        self.opt_calc_collapse = opt_calc_collapse
        self.opt_calc_hlt = opt_calc_hlt
        self.opt_calc_pbt = opt_calc_pbt
        self.opt_calc_wlk = opt_calc_wlk
        self.opt_calc_un = opt_calc_un
        self.opt_calc_hf = opt_calc_hf
        self.opt_calc_cm = opt_calc_cm
        self.opt_calc_gg = opt_calc_gg
        self.opt_calc_box = opt_calc_box
        self.opt_fracrepn = opt_fracrepn
        self.opt_orthu = opt_orthu
        self.opt_uniforce = opt_uniforce
        self.opt_cmwarn = opt_cmwarn