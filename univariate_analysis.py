import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_df_distributions(df, num_bins=30):
    """
    Plot the distribution of variables in a pandas DataFrame using Matplotlib.

    Parameters:
    -----------
    df: pandas.DataFrame
        A DataFrame containing variables to be plotted.
    num_bins: int, optional (default=30)
        The number of bins to use in the histogram.

    Returns:
    --------
    None
    """
    # Iterate over columns and create histograms
    for col in df.columns:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=num_bins)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        plt.show()

def plot_single_distribution(df, column):
    """
    Plot insights about a variable distribution using Seaborn.

    Parameters:
    -----------
    df: pandas.DataFrame
        A DataFrame containing the variable to be plotted.
    column: str
        The name of the variable to plot.

    Returns:
    --------
    None
    """
    # Create a subset of the DataFrame with the selected variable
    plot_data = df[column]

    # Compute descriptive statistics
    mean = plot_data.mean()
    median = plot_data.median()
    skewness = plot_data.skew()
    kurtosis = plot_data.kurtosis()

    # Plot the distribution using Seaborn
    sns.set(style="whitegrid")
    ax = sns.distplot(plot_data)

    # Add vertical lines for mean and median
    ax.axvline(mean, color='r', linestyle='--')
    ax.axvline(median, color='g', linestyle='-')

    # Add text annotations for skewness and kurtosis
    ax.text(0.05, 0.95, f"Skewness: {skewness:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.90, f"Kurtosis: {kurtosis:.2f}", transform=ax.transAxes)

    # Set plot title and axis labels
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    
    # Display the plot
    plt.show()

def plot_boxplot(df, vars_to_plot):
    """
    Compute and display boxplots of selected variables in a pandas DataFrame using Seaborn.

    Parameters:
    -----------
    df: pandas.DataFrame
        A DataFrame containing the variables to be plotted.
    vars_to_plot: list of str
        A list of variable names to plot.

    Returns:
    --------
    None
    """
    # Create a subset of the DataFrame with the selected variables
    plot_data = df[vars_to_plot]

    # Compute and display boxplots using Seaborn
    sns.set(style="ticks", palette="pastel")
    sns.boxplot(data=plot_data, orient="h")
    sns.despine(offset=10, trim=True)


def plot_main_graphs(df, variable):
        """
    Plot the distribution of a given variable in a dataframe using barplot, boxplot, and KDE plot.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the variable to be plotted.
    variable : str
        The name of the variable in the dataframe that needs to be plotted.

    Returns:
    --------
    None
        This function displays the plots directly using matplotlib and seaborn libraries.
    """
    # Plotting barplot
    plt.figure(figsize=(10, 5))
    sns.countplot(x=variable, data=df)
    plt.title(f"Distribution of {variable}")
    plt.show()

    # Plotting boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=variable, data=df)
    plt.title(f"Boxplot of {variable}")
    plt.show()

    # Plotting KDE plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[variable], shade=True)
    plt.title(f"Density plot of {variable}")
    plt.show()

def plot_cdf(data, variable):
    """
    Plots the cumulative distribution function (CDF) of a given variable in a dataframe.
    
    Parameters:
    data (DataFrame): The dataframe containing the variable to plot.
    variable (str): The name of the variable to plot the CDF for.
    
    Returns:
    None
    """
    # Get the values of the variable
    values = data[variable].sort_values()
    
    # Calculate the CDF
    cdf = (1.0 * np.arange(len(values))) / (len(values) - 1)
    
    # Plot the CDF
    sns.lineplot(x=values, y=cdf)
    plt.title("Cumulative Distribution Function of " + variable)
    plt.xlabel(variable)
    plt.ylabel("Cumulative Probability")
    plt.show()