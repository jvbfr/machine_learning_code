import pandas as pd
import numpy as np
import scipy.stats as ss

def test_two_proportions(prop1, prop2, n1, n2, alpha=0.05):
    """
    Perform hypothesis testing to test wether two proportions are statistically different 
    
    Parameters:
        prop1 (float): The proportion of successes in the first sample.
        prop2 (float): The proportion of successes in the second sample.
        n1 (int): The size of the first sample.
        n2 (int): The size of the second sample.
        alpha (float): The significance level of the test (default is 0.05).
    
    Returns:
        (result, p_value): A tuple containing the result of the test (True if the null hypothesis is rejected, 
        False otherwise) and the p-value of the test.
    """
    p = (prop1*n1 + prop2*n2) / (n1 + n2)
    z_score = (prop1 - prop2) / np.sqrt(p*(1-p)*(1/n1 + 1/n2))
    p_value = 2 * (1 - ss.norm.cdf(abs(z_score)))
    result = p_value < alpha
    
    return (result, p_value)