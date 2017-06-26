from scipy import special
from scipy.stats import norm
import math

def probf(fcrit, df1, df2, noncen):
    """PROBF calculates Pr(FCRIT < F(df1,df2,noncen)) using one of four
       methods. The first, most common method uses the cumulative
       distribution function of the non-central F.  If the CDF method will
       fail, the second method uses the Tiku approximation to the non-* central F (Johnson and Kotz).
       In situations where the TIKU will fail or be inaccurate, we use a Normal approximation (from _PROBF).

    :param fcrit: Critical value of F distribution under null hypothesis
    :param df1: Numerator (hypothesis) degrees of freedom
    :param df2: Denominator (error) degrees of freedom
    :param noncen: Noncentrality parameter
    :return:  returns a tuple (prob, fmethod)
              prob, Probability that variable distributed F(df1, df2, noncen)
               will take a value <= fcrit
              fmethod,
               =1, CDF function (no approximation)
               =2, Tiku approximation (best approximation)
               =3, Normal approximation, |Z-score| < 6 (worst
                       approximation)
               =4, Normal approximation, |Z-score| > 6 (approximation
                       but power is almost certainly zero or one)
               =5, Power missing
    """
    if ((df1 < 10**4.4 
         and df2 < 10**5.4 
         and noncen < 10**6.4)
        or
        (df1 < 10**6 
         and df2 < 10 
         and noncen < 10**6)):
        fmethod, prob = _nonadjusted(df1, df2, fcrit, noncen)
    elif (1 <= df1 < 10**9.2 
          and 10 ** 0.6 <= df2 < 10**9.2
          and noncen < 10**6.4):
        fmethod, prob = _tiku_approximation(df1, df2, fcrit, noncen)
    else:
        zscore = _get_zscore(df1, df2, fcrit, noncen)
        fmethod, prob = _normal_approximation(zscore)
    return prob, fmethod


def _normal_approximation(zscore):
    """Normal approximation, value dependent on zscore"""
    if math.fabs(zscore) < 6:
        fmethod = 3
        prob = norm.cdf(zscore)
    else:
        fmethod = 4
        if zscore < -6:
            prob = 0
        elif zscore > 6:
            prob = 1
    return fmethod, prob


def _get_zscore(df1, df2, fcrit, noncen):
    """Calculate zscore for Normal approximation"""
    p1 = 1 / 3
    p2 = -2
    p3 = 1 / 2
    p4 = 2 / 3
    arg1 = ((df1 * fcrit) / (df1 + noncen))
    arg2 = (2 / 9) * (df1 + 2 * noncen) * ((df1 + noncen) ** p2)
    arg3 = (2 / 9) * (1 / df2)
    numz = (arg1 ** p1) - (arg3 * (arg1 ** p1)) - (1 - arg2)
    denz = (arg2 + arg3 * arg1**p4) ** p3
    zscore = numz / denz
    return zscore


def _tiku_approximation(df1, df2, fcrit, noncen):
    """Tiku approximation (best approximation)"""
    h_tiku = 2 * (df1 + noncen)**3 + 3 * (df1 + noncen) * (df1 + 2 * noncen) * (df2 - 2) + (df1 + 3 * noncen) * (df2 - 2)**2
    k_tiku = (df1 + noncen)**2 + (df2 - 2) * (df1 + 2 * noncen)
    df1_tiku = math.floor(0.5 * (df2 - 2) * ((h_tiku**2 / (h_tiku**2 - 4 * k_tiku**3))**0.5 - 1))
    c_tiku = (df1_tiku / df1) / (2 * df1_tiku + df2 - 2) * (h_tiku / k_tiku)
    b_tiku = - df2 / (df2 - 2) * (c_tiku - 1 - noncen / df1)
    fcrit_tiku = (fcrit - b_tiku) / c_tiku
    prob = special.ncfdtr(df1_tiku, df2, 0, fcrit_tiku)
    fmethod = 2
    return fmethod, prob


def _nonadjusted(df1, df2, fcrit, noncen):
    """CDF function (no approximation)"""
    prob = special.ncfdtr(df1, df2, noncen, fcrit)
    fmethod = 1
    return fmethod, prob