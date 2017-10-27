import numpy as np
from glmpowercalc.ranksymm import ranksymm


def power():

    # define initial parameters for power calculation
    xpx = (essencex.T * essencex + (essencex.T * essencex).T) / 2
    cpc = (c_matrix * c_matrix.T + (c_matrix * c_matrix.T).T) / 2
    upu = (u_matrix.T * u_matrix + (u_matrix.T * u_matrix).T) / 2
    rank_X = ranksymm(xpx, tolerance)   #R
    rank_c = ranksymm(cpc, tolerance)
    rank_u = ranksymm(upu, tolerance)
    min_rank_c_u = min(rank_c, rank_u)  #S

    num_col_x = np.shape(essencex)[1]   #Q
    num_row_c = np.shape(c_matrix)[0]   #A
    num_col_u = np.shape(u_matrix)[1]   #B
    num_response = np.shape(beta)[1]    #P