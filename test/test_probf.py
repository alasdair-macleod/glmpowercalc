from unittest import TestCase
from unittest.mock import patch

from glmpowercalc.probf import probf, _normal_approximation, _get_zscore, _tiku_approximation, _nonadjusted


class TestProbf(TestCase):

    @patch('glmpowercalc.probf._nonadjusted')
    def test_probf_nonadjusted(self, mock):
        """probf should call the nonadjusted method for these input values"""
        mock.return_value = None, None
        probf(0, 0, 0, 0)
        assert mock.called

    @patch('glmpowercalc.probf._nonadjusted')
    def test_probf_nonadjusted_path_two(self, mock):
        """probf should call the nonadjusted method for these input values"""
        mock.return_value = None, None
        assert(not mock.called)
        probf(0, 10**5, 9, 10**5.9)
        assert mock.called

    @patch('glmpowercalc.probf._tiku_approximation')
    def test_probf_tiku_approx(self, mock):
        """probf should call the _tiku_approximation method for these input values"""
        mock.return_value = None, None
        assert (not mock.called)
        probf(0, 10**7, 10, 10)
        assert mock.called

    @patch('glmpowercalc.probf._normal_approximation')
    @patch('glmpowercalc.probf._get_zscore')
    def test_probf_norm_approx(self, zmock, mock):
        """probf should call the normal approximation method for these input values"""
        mock.return_value = None, None
        probf(0, 10 ** 9.5, 0, 0)
        assert mock.called

    @patch('scipy.stats.norm.cdf')
    def test__normal_approximation_fmethod_three(self, mock):
        """Should calculate prob using norm.cdf for |zcsore| < 6"""
        expected = (0, 3)
        mock.return_value = 0
        actual = _normal_approximation(0)
        assert mock.called
        assert actual == expected

    @patch('scipy.stats.norm.cdf')
    def test__normal_approximation_fmethod_four(self, mock):
        """Should not call norm.cdf for |zcsore| > 6
        and return prob 0 for -ve values or 1 for +ve values"""
        expected = (0, 4)
        mock.return_value = 0
        actual = _normal_approximation(-7)
        assert not mock.called
        assert actual == expected

        expected = (1, 4)
        mock.return_value = 0
        actual = _normal_approximation(7)
        assert not mock.called
        assert actual == expected

    # TODO raise our own error here and handle sensibly
    @patch('scipy.stats.norm.cdf')
    def test_normal_approximation_method_four(self, mock):
        """Should raise an informative error when zscore = 6"""
        expected = (-1, None)
        with self.assertRaises(UnboundLocalError): # Let's define our own exception types here'
            actual = _normal_approximation(6)
            self.assertEquals(actual, expected)

    def test_get_zscore(self):
        """Should return the expected value for normal data"""
        expected = -0.39007506867134967
        actual = _get_zscore(1, 1, 1, 1)
        self.assertEquals(expected, actual)

    # TODO raise our own error here and handle sensibly
    def test_get_zscore_div_zero(self):
        """Should raise informative exception for values that would result in div/0 error"""
        with self.assertRaises(ZeroDivisionError):
            actual = _get_zscore(0, 1, 1, 0)
        with self.assertRaises(ZeroDivisionError):
            actual = _get_zscore(1, 0, 1, 1)

    @patch('scipy.special.ncfdtr')
    def test__tiku_approximation(self, mock):
        """Should return fmethod 2 and call special.ncfdtr for normal input"""
        expected = (2, 0)
        mock.return_value = 0
        assert not mock.called
        actual = _tiku_approximation(1, -2, 2, 0)
        self.assertEquals(expected, actual)

    # TODO raise our own error here and handle sensibly
    @patch('scipy.special.ncfdtr')
    def test__tiku_approximation(self, mock):
        """Should return fmethod 2 and call special.ncfdtr for normal input"""
        mock.return_value = 0
        assert not mock.called
        with self.assertRaises(ZeroDivisionError):
            actual = _tiku_approximation(0, 0, 0, 0)


    @patch('scipy.special.ncfdtr')
    def test__nonadjusted(self, mock):
        """Should calculate prob using special.ncfdtr and return fmethod 1"""
        expected = (0, 1)
        mock.return_value = 0
        actual = _nonadjusted(0, 0, 0, 0)
        self.assertEquals(expected, actual)


