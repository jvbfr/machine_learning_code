import pandas as pd
import numpy as np

def get_summary(data):
    """
    Generates a summary of numerical attributes in a pandas DataFrame. The summary includes:

    - attributes: the column names of the numerical attributes
    - min: the minimum value in each attribute
    - max: the maximum value in each attribute
    - range: the range (difference between max and min) in each attribute
    - Q1: the first quartile of each attribute
    - median: the median value of each attribute
    - Q3: the third quartile of each attribute
    - IQR: the interquartile range (difference between Q3 and Q1) of each attribute
    - mean: the mean value of each attribute
    - std: the standard deviation of each attribute
    - skew: the skewness of each attribute
    - kurtosis: the kurtosis of each attribute

    Parameters:
    -----------
    data: pandas DataFrame
        The DataFrame containing features

    Returns:
    --------
    summary: pandas DataFrame
        The summary of numerical attributes
    """

    numerical_attributes = data.select_dtypes( include = [ 'int64', 'float64'] )
    # Central Tendency - mean, median
    ct1 = pd.DataFrame(numerical_attributes.apply(np.mean)).T
    ct2 = pd.DataFrame(numerical_attributes.apply(np.median)).T

    # Dispersion - std, min, max, range, skew, kurtosis, Q1, Q3, IQR
    d1 = pd.DataFrame(numerical_attributes.apply(np.std)).T 
    d2 = pd.DataFrame(numerical_attributes.apply(min)).T 
    d3 = pd.DataFrame(numerical_attributes.apply(max)).T 
    d4 = pd.DataFrame(numerical_attributes.apply(lambda x: x.max() - x.min())).T 
    d5 = pd.DataFrame(numerical_attributes.apply(lambda x: x.skew())).T 
    d6 = pd.DataFrame(numerical_attributes.apply(lambda x: x.kurtosis())).T
    d7 = pd.DataFrame(numerical_attributes.apply(lambda x: np.quantile(x, 0.25))).T
    d9 = pd.DataFrame(numerical_attributes.apply(lambda x: np.quantile(x, 0.75))).T
    d10 = pd.DataFrame(numerical_attributes.apply(lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25))).T

    # concatenate
    m = pd.concat([d2, d3, d4, d7, ct2, d9, d10, ct1, d1, d5, d6]).T
    m.columns = ['min', 'max', 'range', 'Q1', 'median', 'Q3', 'IQR', 'mean', 'std', 'skew', 'kurtosis']

    return m

def main_view(data):

    shape = data.shape
    nans = data.isna().mean()
    stats = get_summary(data)

    return shape, nans, stats


