from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

def stratified_cross_validation(model, X, y, n_splits=5):
        """
    Perform stratified cross-validation on a given machine learning model using the 
    provided data and return the classification reports for each fold.
    
    Parameters:
    -----------
    model: sklearn estimator object
        The machine learning model to be evaluated.
        
    X: array-like of shape (n_samples, n_features)
        The input data.
        
    y: array-like of shape (n_samples,)
        The target variable for the input data.
        
    n_splits: int, optional (default=5)
        The number of folds to create for cross-validation.
        
    Returns:
    --------
    reports: list
        A list of dictionaries containing the classification report for each fold.
        The dictionary has the following keys: 'precision', 'recall', 'f1-score',
        and 'support'. The value for each key is a dictionary containing the metric
        value for each class, as well as the weighted average of the metric.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    reports = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Calculate the classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Append the score and report to the lists
        scores.append(report['weighted avg']['f1-score'])
        reports.append(report)

    # Calculate the mean score
    mean_score = sum(scores) / len(scores)

    # Print the mean score and classification report
    print(f"Mean F1-Score: {mean_score:.3f}")
    print(classification_report(y, model.predict(X), digits=3))
    
    return reports
