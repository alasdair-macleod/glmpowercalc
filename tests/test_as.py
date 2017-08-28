from glmpowercalc.as_fn import l20, l30, l60, l260_finish
from unittest import TestCase
from mock import patch

class TestAs(TestCase):

    @patch('glmpowercalc.as_fn.l60')
    def test_l20_straigTol60(self, mock):
        """
        If index j is greater than ir,

        ir is definrs as 'the number of chi squared terms in the sum'
        //TODO This is the size of qWeight, which we still need to understand.
        """
        l20(n=[1,2,3], j=2, alb=[0,0,0],anc=[0,0,0],ir=0,sd=0,amean=0,almax=0)
        self.assertTrue(mock.called)

    @patch('glmpowercalc.as_fn.l30')
    def test_l20_straigTol60(self, mock):
        """
        If index j is less than ir,

        If we have both df and non-central parameters larger or equal to 0, then run l30
        We take the jth element of the following vectors and if any of these are
        ALB, IRRx1 vector of constant multipliers
        ANC, Vector of noncentrality parameters
        N, Vector of degrees of freedom
        """
        l20(n=[1,2,3], j=1, alb=[0,0,0],anc=[0,0,0],ir=3,sd=0,amean=0,almax=0)
        mock.assert_called_with(alj=0, almax=0, amean=0, ancj=0, nj=1, sd=0)\

    @patch('glmpowercalc.as_fn.l260_finish')
    def test_l20_straigTol60(self, mock):
        """
        If index j is less than ir,

        If we have df or non-central parameters smaller than 0, then run l260_finish loop
        """
        l20(n=[-1,2,3], j=1, alb=[0,0,0],anc=[0,0,0],ir=3,sd=0,amean=0,almax=0)
        self.assertTrue(mock.called)



