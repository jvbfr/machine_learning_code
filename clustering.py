from sklearn.feature_selection import SelectKBest, f_classif

def rank_features(X, y, n_features):
    """
    Rank the best features for clustering using ANOVA F-value.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target variable.

    n_features : int
        The number of features to select.

    Returns
    -------
    feature_ranks : list of tuples
        A list of tuples, where each tuple contains the index of the feature and its corresponding score.
    """
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    scores = selector.scores_
    feature_ranks = [(i, score) for i, score in enumerate(scores)]
    feature_ranks.sort(key=lambda x: x[1], reverse=True)
    return feature_ranks