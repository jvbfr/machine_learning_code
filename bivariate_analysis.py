import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss

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

def plot_cramer_v(df):
    """
    Compute the Cramer's V statistic for each pair of categorical variables in a DataFrame and display the results
    as a Seaborn heatmap.

    Parameters:
    -----------
    df: pandas.DataFrame
        A DataFrame containing categorical variables.

    Returns:
    --------
    None
    """
    # Create a DataFrame to store the Cramer's V results
    columns = df.select_dtypes(include='category').columns
    results = pd.DataFrame(np.zeros((len(columns), len(columns))), columns=columns, index=columns)

    # Compute Cramer's V for each pair of categorical variables
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i < j:
                results.loc[col1, col2] = cramer_v(df[col1], df[col2])
    
    # Plot the results using Seaborn
    sns.set(style='white')
    mask = np.triu(np.ones_like(results, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(results, mask=mask, cmap=cmap, vmin=0, vmax=1, center=.5,
                     square=True, annot=True, fmt='.2f', linewidths=.5, cbar_kws={"shrink": .5})
    
    # Set plot title and axis labels
    plt.title("Cramer's V Correlation Matrix")
    plt.xlabel("")
    plt.ylabel("")

    # Display the plot
    plt.show()


def plot_correlation_matrix(df):

    """
    Create a correlation matrix between numerical variables in a DataFrame and plot it using Seaborn.

    Parameters:
    -----------
    df: pandas.DataFrame
        A DataFrame containing numerical variables.

    Returns:
    --------
    None
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Plot the correlation matrix using Seaborn
    sns.set(style="white")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Set plot title and axis labels
    plt.title("Correlation Matrix")
    plt.xlabel("")
    plt.ylabel("")
    
    # Display the plot
    plt.show()

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


def plot_scatter(df, x_col, y_col):
    """
    Visualize the relationship between two numerical variables in a dataframe using a scatter plot.

    Args:
    df (pandas.DataFrame): The dataframe containing the data to plot.
    x_col (str): The name of the column containing the x-axis variable.
    y_col (str): The name of the column containing the y-axis variable.

    Returns:
    None
    """
    sns.scatterplot(x=x_col, y=y_col, data=df)


def plot_polynomial(df, x_col, y_col, order=2):
    """
    Visualize the non-linear relationship between two variables in a dataframe using a scatter plot with
    a polynomial regression line.

    Args:
    df (pandas.DataFrame): The dataframe containing the data to plot.
    x_col (str): The name of the column containing the x-axis variable.
    y_col (str): The name of the column containing the y-axis variable.
    order (int): The order of the polynomial regression line. Default is 2.

    Returns:
    None
    """
    sns.lmplot(x=x_col, y=y_col, data=df, order=order)