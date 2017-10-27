from unittest import TestCase
from glmpowercalc.UserInput import Input


class TestUserInput(TestCase):
    def testInput(self):
        test = Input(essencex=np.matrix([[1]]), sigma=np.matrix([[1]]), beta=np.matrix([[1]]), c_matrix=np.matrix([[1]]))
