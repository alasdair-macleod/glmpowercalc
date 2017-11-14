class Constants:
    # UNIREP test statistic CDF calculation Methods
    UCDF_MULLER1989_APPROXIMATION = 'Muller and Barton (1989) approximation'
    UCDF_MULLER2004_APPROXIMATION = 'Muller, Edwards and Taylor (2004) approximation'
    UCDF_EXACT_DAVIES = 'Muller, Edwards and Taylor (2004) exact, via Davies algorithm'
    UCDF_EXACT_DAVIES_FAIL = 'If Davies fails, return Muller2004 approximation'

    # mean epsilon hat and mean epsilon tilde approximation methods
    EPSILON_MULLER1989 = 'Muller and Barton (1989) approximation'
    EPSILON_MULLER2004 = 'Method 1, Muller, Edwards, and Taylor (2004)'

    # MULTIREP test statistic CDF calculation Methods
    MULTI_HLT_PILLAI = '1 Pillai (1954, 55) 1 moment null approx'
    MULTI_HLT_MCKEON = '2 McKeon (1974) two moment null approx'
    MULTI_HLT_PILLAI_OS = '3 Pillai (1959) one moment null approx+ OS noncen mult'
    MULTI_HLT_MCKEON_OS = '4 McKeon (1974) two moment null approx+ OS noncen mult'

    MULTI_PBT_PILLAI = '1 Pillai (1954, 55) one moment null approx'
    MULTI_PBT_MULLER = '2 Muller (1998) two moment null approx'
    MULTI_PBT_PILLAI_OS = '3 Pillai (1959) one moment null approx + OS noncen mult'
    MULTI_PBT_MULLER_OS = '4 Muller (1998) two moment null approx + OS noncen mult'
    
    MULTI_WLK_RAO = 'Rao (1951) two moment null approx'
    MULTI_WLK_RAO_OS = 'Rao (1951) two moment null approx + OS noncen mult'

    #CL_TYPE
    CLTYPE_DESIRED_KNOWN = '1 BETA known, SIGMA estimated and CL desired'
    CLTYPE_DESIRED_ESTIMATE = '2 BETA estimated, SIGMA estimated and CL desired'
    CLTYPE_NOT_DESIRED = 'Confidence intervals for power not desired'

    #FMETHOD
    FMETHOD_NOAPPROXIMATION = '1 CDF function (no approximation)'
    FMETHOD_TIKU = '2 Tiku approximation (best approximation)'
    FMETHOD_NORMAL_SM = '3 Normal approximation, |Z-score| < 6 (worst approximation)'
    FMETHOD_NORMAL_LR = '4 Normal approximation, |Z-score| > 6 (approximation but power is almost certainly 0 or 1)'
    FMETHOD_MISSING = '5 Power missing'
