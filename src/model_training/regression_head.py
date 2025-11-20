"""Utility functions for regression heads for text/non-text features.
"""
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score


def optimise_regression_model(model, X, y, cv=5, scoring='neg_root_mean_squared_error'):
    # Define hyperparameter grids for supported regression models
    if isinstance(model, Ridge):
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }
    elif isinstance(model, RandomForestRegressor):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif isinstance(model, GradientBoostingRegressor):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif isinstance(model, SVR):
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    elif isinstance(model, KNeighborsRegressor):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    elif isinstance(model, DummyRegressor):
        param_grid = {}
    else:
        raise ValueError("Model type not supported for regression optimization.")

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,  # e.g., 'neg_root_mean_squared_error'
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_ if 'neg_' in scoring else grid_search.best_score_

    return best_model, best_params, best_score


def evaluate_regression_model(model, X, y, cv=5, scoring='neg_root_mean_squared_error'):
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    avg_score = -scores.mean() if 'neg_' in scoring else scores.mean()

    model.fit(X, y)
    return model, model.get_params(), avg_score
