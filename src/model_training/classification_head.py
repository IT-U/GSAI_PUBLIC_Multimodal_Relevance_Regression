"""Utility functions for classification heads for text/non-text features.
"""
import numpy as np
# import xgboost as xgb
from typing import Tuple, Dict, Any, TypeVar
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Create a generic type variable for models
T = TypeVar('T')


def optimise_model(model: T, X: np.ndarray, y: np.ndarray, cv: int = 5,
                   scoring: str = 'f1_macro') -> Tuple[T, Dict[str, Any], float]:
    """Optimises the hyperparameters of a given model using GridSearchCV.

    Args:
        model (T): The machine learning model to be optimized.
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        cv (int, optional): The number of cross-validation folds. Defaults to 5.
        scoring (str, optional): The scoring metric to use. Defaults to 'f1_macro'.

    Raises:
        ValueError: If the model type is not supported for hyperparameter optimization.

    Returns:
        Tuple[T, Dict[str, Any], float]: A tuple containing the best model, the best hyperparameters, and the best score.
    """
    # Define default hyperparameter grids for supported models
    if isinstance(model, LogisticRegression):
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],  # 'l1' requires solver adjustments
            'solver': ['lbfgs', 'saga'],
            'max_iter': [1000, 2000]
        }
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
    elif isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    # elif isinstance(model, xgb.XGBClassifier):
    #     param_grid = {
    #         'n_estimators': [50, 100, 200],
    #         'max_depth': [3, 5, 7],
    #         'learning_rate': [0.01, 0.1, 0.2]
    #     }
    elif isinstance(model, GradientBoostingClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    elif isinstance(model, MLPClassifier):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'max_iter': [500]
        },
    elif isinstance(model, GaussianNB):
        param_grid = {
            'var_smoothing': [1e-09, 1e-08, 1e-07]
        }
    else:
        raise ValueError("Model type not supported for hyperparameter optimization.")

    # Set up cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5,
                   scoring: str = 'f1_macro') -> Tuple[Any, Dict[str, Any], float]:
    """
    Evaluates a pre-defined model using cross-validation.

    Args:
        model (Any): The machine learning model to be evaluated.
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        cv (int, optional): The number of cross-validation folds. Defaults to 5.
        scoring (str, optional): The scoring metric to use. Defaults to 'f1_macro'.

    Returns:
        Tuple[Any, Dict[str, Any], float]: A tuple containing the model,
        its parameters (as a dictionary), and the average cross-validation score.
    """
    # Set up cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Evaluate the model using cross_val_score
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    average_score = scores.mean()
    
    # Fit the model on the entire dataset after evaluation
    model.fit(X, y)

    # Retrieve the model's current parameters
    params = model.get_params()

    return model, params, average_score