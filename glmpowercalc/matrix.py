import numpy as np

class Matrix(object):
    label = None
    matrix = None

    def __init__(self, label, data):
        self.label=label
        self.matrix = np.matrix(data)

###################
def createMatrix():
    a = Matrix('M', [[1, 2], [3, 4]])
    print("Matrix is called {0}, and has data {1}".format(a.label, a.matrix))


