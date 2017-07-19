import numpy as np

class Matrix(object):
    label = None
    matrix = None

    def __init__(self, label, data):
        self.label=label
        self.matrix = np.matrix(data)
