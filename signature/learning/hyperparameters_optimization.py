from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from tqdm import tqdm


def tscv_loop(X, y, model, n_splits: int = 5, burn_in: int = 0):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mse = []

    # 2. Cross-Validation Loop
    for train_idx, val_idx in tscv.split(X[burn_in:]):
        # Split data
        X_tr, X_val = X[:burn_in + train_idx[-1] + 1], X[:burn_in + val_idx[-1] + 1] # Full path for state continuity
        y_tr, y_val = y[train_idx], y[val_idx]

        # Clone to ensure a clean fit for every fold
        m = clone(model)
        m.fit(X_tr, y_tr)

        # Predict only on the validation segment
        # We slice the prediction to match the y_val indices
        y_pred = m.predict(X_val)[-len(y_val):]

        # TODO: write for generic metrics?
        mse = mean_squared_error(y_val, y_pred)
        cv_mse.append(mse)

    avg_mse = np.mean(cv_mse)
    return avg_mse


def grid_search_cv_efm(X, y, model, param_grid, burn_in, n_splits=5):
    """
    Performs a Grid Search over lambda and alpha while preserving path continuity.
    """
    grid = ParameterGrid(param_grid)
    results = []
    for params in tqdm(grid):
        # Update model parameters
        model.set_params(**params)

        # Time series cross-validation loop
        avg_mse = tscv_loop(model=model, X=X, y=y, burn_in=burn_in, n_splits=n_splits)

        results.append({**params, 'mse': avg_mse})
        print(f"Params: {params} | Mean CV MSE: {avg_mse:.6f}")

    # Convert to DataFrame and find best
    results_df = pd.DataFrame(results)
    best_params = grid[results_df['mse'].idxmin()]
    return best_params, results_df

def grid_search_efm(X, y, model, param_grid, burn_in):
    """
    Performs a Grid Search over lambda and alpha while preserving path continuity.
    """
    grid = ParameterGrid(param_grid)
    results = []
    for params in grid:
        # Update model parameters
        model.set_params(**params)

        # Time series cross-validation loop
        m = clone(model)
        m.fit(X, y)
        y_pred = m.predict(X)[-len(y):]

        # TODO: write for generic metrics?
        mse = mean_squared_error(y, y_pred)

        results.append({**params, 'mse': mse})
        print(f"Params: {params} | RMSE: {np.sqrt(mse):.6f}")

    # Convert to DataFrame and find best
    results_df = pd.DataFrame(results)
    best_params = grid[results_df['mse'].idxmin()]
    return best_params, results_df


def optimize_lam_cv(X, y, model, burn_in, bounds, n_splits=5, init_guess = None):
    def loss(x):
        # Update lambda in the transformer
        model.set_params(sig__lam=x)

        avg_mse = tscv_loop(model=model, X=X, y=y, burn_in=burn_in, n_splits=n_splits)
        print(f"Testing lam: {x} | CV MSE: {avg_mse:.6f}")
        return avg_mse

    # Run Optimizer
    # Powell is good for non-differentiable or noisy objective functions
    dim = len(bounds)
    if init_guess is None:
        init_guess = [0.5 * (bound[0] + bound[1]) for bound in bounds]
    res = minimize(
        loss,
        x0=init_guess,
        bounds=bounds,
        method='Powell',
        options={'xtol': 1e-3, 'ftol': 1e-3}
    )
    print(res)
    return res.x