<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np

def ranksymm(Matrix, tolerance):
=======
=======
>>>>>>> dd39514e76f6b582b4ceb09793906eb0e63a16a7
import Matrix from matrix;

def ranksymm(matrix, matrixlbl, tolerance):
>>>>>>> dd39514e76f6b582b4ceb09793906eb0e63a16a7
	"""This function computes the rank of a square symmetric       
           nonnegative definite matrix via eigenvalues.
                                                            
       :param Matrix: Class define by Matrix(), with matrix and label
       :param tolerance:  value not tolerated, numeric zero (global)
                                                
       :return rankmatrix: 
           = . if MATRIX is not symmetric or positive definite
           = rank of the matrix
    """
    maxabsval = abs(Matrix.matrix).max()
    nmatrix = Matrix.matrix / maxabsval
    evals = np.linalg.eigvals(Matrix.matrix)
    
    # empty matrix
    if Matrix.matrix.shape[1] = 0:
    	print("ERROR 55: Matrix ", Matrix.label, "does not exist.")
    
    # number of rows not equal to number of columns 
    elif Matrix.matrix.shape[0] != Matrix.matrix.shape[1]:
    	print("ERROR 56: Matrix ", Matrix.label, "is not square.")
    
    #TODO 
    # matrix with all missing values
    #elif (Matrix.matrix == np.NA).all():
    #	print("ERROR 57: Matrix ", Matrix.label, "is all missing values")
    
    # matrix with all zero
    elif maxabsval == 0:
    	print("ERROR 58: Matrix ", Matrix.label, "has MAX(ABS(all elements)) = exact zero.")

    # matrix not symmetric
    elif abs(Matrix.matrix - Matrix.matrix.T).max() >= tolerance**0.5:
    	print("ERROR 59: Matrix ", Matrix.label, "is not symmetric within sqrt(tolerance).")

    # matrix not nonnegative definite 
    elif evals.min() < -tolerance**0.5:
    	print("ERROR 60: Matrix ", Matrix.label, " is *NOT* nonnegative definite (and has at ",
                                                        "least one eigenvalue strictly less than ",
                                                        "zero). This may happen due to programming ",
                                                        "error or rounding error of a nearly LTFR ",
                                                        "matrix. This may be able to be fixed using ",
                                                        "usual scaling/centering techniques. The ",
                                                        "Eigenvalues/MAX(ABS(original matrix)) are: ", evals)
    	print(" The max(abs(original matrix)) is ", maxbasval)

    else:
    rankmatrix = sum(evals >= tolerance)

    return rankmatrix



