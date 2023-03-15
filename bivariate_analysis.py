import numpy as np
from scipy import stats  as ss

def cramer_v( x, y ):
        """
    Calculate Cramer's V statistic for two categorical variables.

    Parameters:
    -----------
    x: pandas.Series
        A categorical variable represented as a pandas.Series.
    y: pandas.Series
        A categorical variable represented as a pandas.Series.

    Returns:
    --------
    float
        The Cramer's V statistic for the two variables.

    Notes:
    ------
    Cramer's V statistic is a measure of association between two categorical variables.
    It ranges between 0 (no association) and 1 (perfect association). This implementation
    uses a corrected version of the chi-squared test and takes into account the number
    of rows and columns of the contingency table to correct for bias.
    """
    cm = pd.crosstab( x, y ).as_matrix()
    n = cm.sum()
    r, k = cm.shape
    
    chi2 = ss.chi2_contingency( cm )[0]
    chi2corr = max( 0, chi2 - (k-1)*(r-1)/(n-1) )
    
    kcorr = k - (k-1)**2/(n-1)
    rcorr = r - (r-1)**2/(n-1)
    
    return np.sqrt( (chi2corr/n) / ( min( kcorr-1, rcorr-1 ) ) )