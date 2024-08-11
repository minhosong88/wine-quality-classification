from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from custom_logistic_regression import MultiClassLogisticRegression


def train_and_evaluate(X, y):
    """Train and evaluate the logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2)

    param_range = {'C': [
        10**i for i in range(-4, 5)], 'solver': [None, "hessian", "stochastic", "steepest"]}

    best_accuracy = 0
    best_params = {}

    for C in param_range['C']:
        for solver in param_range['solver']:
            mclr = MultiClassLogisticRegression(
                eta=0.1, C=C, solver=solver, iterations=140)
            mclr.fit(X_train, y_train)
            y_hat = mclr.predict(X_test)
            accuracy = accuracy_score(y_test, y_hat)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'C': C, 'solver': solver}

    return best_params, best_accuracy
