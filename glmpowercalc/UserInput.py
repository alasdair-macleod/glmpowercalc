import numpy as np


class Input:
    def __init__(self, essencex, sigma, beta, c_matrix,
                 u_matrix=None, theta=None, repn=None, betascal=None, sigscal=None, rhoscal=None, alpha=None, round=None,
                 tolerance=None, ucdf=None, umethod=None, mmethod=None, ip_plan=None, n_ip=None, rank_ip=None,
                 limcase=None, cltype=None, n_est=None, rank_est=None, alpha_cl=None, alpha_cu=None, sigtype=None,
                 UnirepUncorrected=None, UnirepHuynhFeldt=None, UnirepHuynhFeldtChiMuller=None,
                 UnirepGeisserGreenhouse=None, UnirepBox=None, EpsilonAppHuynhFeldt=None,
                 EpsilonAppHuynhFeldtChiMuller=None, EpsilonAppGeisserGreenhouse=None, MultirepHLT=None, MultiPBT=None,
                 MultiWLK=None):
        """

        :param essencex:
        :param sigma:
        :param beta:
        :param c_matrix:
        :param u_matrix:
        :param theta:
        :param repn:
        :param betascal:
        :param sigscal:
        :param rhoscal:
        :param alpha:
        :param round:
        :param tolerance:
        :param ucdf:
        :param umethod:
        :param mmethod:
        :param ip_plan:
        :param n_ip:
        :param rank_ip:
        :param limcase:
        :param cltype:
        :param n_est:
        :param rank_est:
        :param alpha_cl:
        :param alpha_cu:
        :param sigtype:
        :param UnirepUncorrected:
        :param UnirepHuynhFeldt:
        :param UnirepHuynhFeldtChiMuller:
        :param UnirepGeisserGreenhouse:
        :param UnirepBox:
        :param EpsilonAppHuynhFeldt:
        :param EpsilonAppHuynhFeldtChiMuller:
        :param EpsilonAppGeisserGreenhouse:
        :param MultirepHLT:
        :param MultiPBT:
        :param MultiWLK:
        """

        # A: check for required input matrices
        if c_matrix is None:
            raise Exception("ERROR 2: The matrix C has not been supplied.")
        else:
            self.c_matrix = c_matrix

        if beta is None:
            raise Exception("ERROR 3: The matrix BETA has not been supplied.")
        else:
            self.beta = beta

        if sigma is None:
            raise Exception("ERROR 4: The matrix SIGMA has not been supplied.")
        else:
            self.sigma = sigma

        if essencex is None:
            raise Exception("ERROR 5: The matrix ESSENCEX has not been supplied.")
        else:
            self.essencex = essencex

        # B: define default matrices
        if u_matrix is None:
            self.u_matrix = np.matrix(np.ones(np.shape(self.beta)[1]))
        else:
            self.u_matrix = u_matrix

        if theta is None:
            # TODO array or matrix
            self.theta = np.zeros((np.shape(self.c_matrix)[0], np.shape(self.u_matrix)[1]))
        else:
            self.theta = theta

        if repn is None:
            self.repn = 1
        else:
            self.repn = repn

        if betascal is None:
            # TODO array or matrix or scale
            self.betascal = np.matrix([[1]])
        else:
            self.betascal = betascal

        if sigscal is None:
            # TODO array or matrix or scale
            self.sigscal = np.matrix([[1]])
        else:
            self.sigscal = sigscal

        if rhoscal is None:
            # TODO array or matrix or scale
            self.rhoscal = np.matrix([[1]])
        else:
            self.rhoscal = rhoscal

        if alpha is None:
            # TODO array or matrix or scale
            self.alpha = np.matrix([[0.05]])
        else:
            self.alpha = alpha

        if round is None:
            self.round = 3
        else:
            self.round = round

        if tolerance is None:
            self.tolerance = 1e-12
        else:
            self.tolerance = tolerance

        if ucdf is None:
            self.ucdf = [2, 2, 2, 2, 2]
        else:
            self.ucdf = ucdf

        if UnirepUncorrected is None:
            self.UnirepUncorrected = 'Two Moment Approximation'
        else:
            self.UnirepUncorrected = UnirepUncorrected

        if UnirepHuynhFeldt is None:
            self.UnirepHuynhFeldt = 'Two Moment Approximation'
        else:
            self.UnirepHuynhFeldt = UnirepHuynhFeldt

        if UnirepHuynhFeldtChiMuller is None:
            self.UnirepHuynhFeldtChiMuller = 'Two Moment Approximation'
        else:
            self.UnirepHuynhFeldtChiMuller = UnirepHuynhFeldtChiMuller

        if UnirepGeisserGreenhouse is None:
            self.UnirepGeisserGreenhouse = 'Two Moment Approximation'
        else:
            self.UnirepGeisserGreenhouse = UnirepGeisserGreenhouse

        if UnirepBox is None:
            self.UnirepBox = 'Two Moment Approximation'
        else:
            self.UnirepBox = UnirepBox

        if umethod is None:
            self.umethod = [2, 2, 2]
        else:
            self.umethod = umethod

        if EpsilonAppHuynhFeldt is None:
            self.EpsilonAppHuynhFeldt = "Muller2007"
        else:
            self.EpsilonAppHuynhFeldt = EpsilonAppHuynhFeldt

        if EpsilonAppHuynhFeldtChiMuller is None:
            self.EpsilonAppHuynhFeldtChiMuller = "Muller2007"
        else:
            self.EpsilonAppHuynhFeldtChiMuller = EpsilonAppHuynhFeldtChiMuller

        if EpsilonAppGeisserGreenhouse is None:
            self.EpsilonAppGeisserGreenhouse = "Muller2007"
        else:
            self.EpsilonAppGeisserGreenhouse = EpsilonAppGeisserGreenhouse

        if mmethod is None:
            self.mmethod = [4, 2, 2]
        else:
            self.mmethod = mmethod

        if MultirepHLT is None:
            self.MultirepHLT = "Mckeon1974"
        else:
            self.MultirepHLT = MultirepHLT

        if MultiPBT is None:
            self.MultiPBT = "Muller1998"
        else:
            self.MultiPBT = MultiPBT

        if MultiWLK is None:
            self.MultiWLK = "Rao"
        else:
            self.MultiWLK = MultiWLK

        if ip_plan is None:
            self.ip_plan = 0
        else:
            self.ip_plan = ip_plan

        if cltype is None:
            self.cltype = -1
        else:
            self.cltype = cltype

        if alpha_cl is None:
            self.alpha_cl = 0.025
        else:
            self.alpha_cl =alpha_cl

        if alpha_cu is None:
            self.alpha_cu = 0.025
        else:
            self.alpha_cu = alpha_cu

        if sigtype is None:
            if self.cltype >= 1:
                self.sigtype = 1
            else:
                self.sigtype = 0
        else:
            self.sigtype = sigtype