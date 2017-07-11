

def ranksymm(matrix, matrixlbl, tolerance):
	"""This function computes the rank of a square symmetric       
       nonnegative definite matrix via eigenvalues.                                                  
                                                            
        :param matrix: matrix which will be checked                  
        :param matrixlbl: label to identify the matrix               
        :param tolerance:  value not tolerated, numeric zero (global)
                                                
        :return rankmatrix: 
            = . if MATRIX is not symmetric or positive definite     
            = rank of the matrix
    """
    