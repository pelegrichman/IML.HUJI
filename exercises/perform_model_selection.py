from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from plotly.graph_objs import Figure
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    def true_poly(x: float, noise_scale: float) -> float:
        """Returns for x the value of:
        (x+3)(x+2)(x+1)(x-1)(x-2) + eps
        """
        eps: float = np.random.normal(loc=0.0, scale=noise_scale, size=1)
        return ((x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)) + eps

    # X = n_samples sampled uniformly from [-1.2, 2].
    X: np.ndarray = np.random.uniform(-1.2, 2, size=n_samples)

    # create True Poly
    sorted_X = np.sort(X)
    noiseless_poly: np.ndarray = np.array([true_poly(x, 0) for x in sorted_X])
    y: np.ndarray = np.array([true_poly(x, noise) for x in X])

    # Split train test
    X_train, y_train, X_test, y_test = split_train_test(DataFrame(X), Series(y.ravel()), train_proportion=.66)

    fig = go.Figure(data=[go.Scatter(x=sorted_X.flatten(), y=noiseless_poly.flatten(), mode="lines+markers",
                                     name='true poly', marker=dict(color="black", opacity=.6), showlegend=True),

                          go.Scatter(x=X_train.iloc[:, 0], y=y_train, mode="markers", name='train',
                                     marker=dict(color='blue', opacity=.6), showlegend=True),

                          go.Scatter(x=X_test.iloc[:, 0], y=y_test, mode="markers", name='test',
                                     marker=dict(color='red', opacity=.6), showlegend=True)])

    fig = fig.update_layout(title={"text": f"(1) Simulated Data noise_level = {noise}"
                                           f" number of samples = {n_samples}"},
                            xaxis={"title": "x - Explanatory Variable"},
                            yaxis={"title": "y - Response"},
                            height=1000,
                            width=2000)

    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_range: List[int] = list(range(0, 11))
    # Mean - Train and validation errors for all values of k
    train_errors, validation_errors = [], []
    for k in k_range:
        model: PolynomialFitting = PolynomialFitting(k)
        train_err, val_error = cross_validate(model, X_train.to_numpy(),
                                              y_train.to_numpy().reshape([-1, 1]),
                                              mean_square_error)
        train_errors.append(train_err), validation_errors.append(val_error)

        print(f"Polynomial degree to fit - Model Complexity: {k} \n"
              f"Average Train Error: {train_err} \n"
              f"Average Validation Error: {val_error} \n\n")

    fig: Figure = go.Figure()
    fig.add_traces(data=[go.Scatter(x=k_range, y=train_errors, mode="markers+lines", name='avg_train_mse',
                                    marker=dict(color='blue'), showlegend=True),

                         go.Scatter(x=k_range, y=validation_errors, mode="markers+lines", name='avg_val_mse',
                                    marker=dict(color='red'), showlegend=True)])

    fig.update_layout(title={"text": f"(2) Average MSE of train and "
                                     f"validation as function of k "
                                     f"noise level: {noise} "
                                     f"number of samples: {n_samples}"},
                      xaxis={"title": "x - k degree of Poly -> model's complexity"},
                      yaxis={"title": "y - MSE error"},
                      height=1000,
                      width=2000)

    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star: int = k_range[np.argmin(validation_errors)]
    fitted_model: PolynomialFitting = PolynomialFitting(k=k_star).fit(X_train.to_numpy(), y_train.to_numpy())
    test_err: float = mean_square_error(y_test.to_numpy(), fitted_model.predict(X_test.to_numpy()))

    print(f"Best Polynomial degree to fit k_star: {k_star} \n"
          f"Fitted Model Test Error: {np.round(test_err, decimals=2)} \n"
          f"Average Validation Error for k_star: {validation_errors[k_star]} \n\n")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)

    # Split train test
    X_train, y_train, X_test, y_test = split_train_test(X, y, n_samples=50)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range: np.ndarray = np.linspace(0.02, 2, n_evaluations).flatten()

    # Mean of - Train and validation errors for all values of lam
    r_train_errors, r_validation_errors = [], []
    l_train_errors, l_validation_errors = [], []
    for lam in lam_range:
        ridge_model, lasso_model = RidgeRegression(lam), Lasso(alpha=lam)

        train_err, val_error = cross_validate(ridge_model, X_train.to_numpy(),
                                              y_train.to_numpy().reshape([-1, 1]),
                                              mean_square_error)
        r_train_errors.append(train_err), r_validation_errors.append(val_error)

        train_err, val_error = cross_validate(lasso_model, X_train.to_numpy(),
                                              y_train.to_numpy().reshape([-1, 1]),
                                              mean_square_error)
        l_train_errors.append(train_err), l_validation_errors.append(val_error)

        # print(f"Regularization Parameter {lam} \n"
        #       f"Average Train Error: {train_err} \n"
        #       f"Average Validation Error: {val_error} \n\n")

    fig: Figure = go.Figure()
    fig.add_traces(data=[go.Scatter(x=lam_range, y=r_train_errors, mode="markers+lines",
                                    name='avg_ridge_train_mse',
                                    marker=dict(color='blue'), showlegend=True),

                         go.Scatter(x=lam_range, y=r_validation_errors, mode="markers+lines",
                                    name='avg_ridge_val_mse',
                                    marker=dict(color='red'), showlegend=True)])

    fig.update_layout(title={"text": f"(7) Average MSE of train and "
                                     f"validation as function of k "
                                     f"Model : Ridge "
                                     f"number of samples: {n_samples}"},
                      xaxis={"title": "x - lambda Regularization parameter"},
                      yaxis={"title": "y - MSE error"},
                      height=1000,
                      width=2000)

    fig.show()

    fig: Figure = go.Figure()
    fig.add_traces(data=[go.Scatter(x=lam_range, y=l_train_errors, mode="markers+lines",
                                    name='avg_lasso_train_mse',
                                    marker=dict(color='blue'), showlegend=True),

                         go.Scatter(x=lam_range, y=l_validation_errors, mode="markers+lines",
                                    name='avg_lasso_val_mse',
                                    marker=dict(color='red'), showlegend=True)])

    fig.update_layout(title={"text": f"(7) Average MSE of train and "
                                     f"validation as function of k "
                                     f"Model : Lasso "
                                     f"number of samples: {n_samples}"},
                      xaxis={"title": "x - lambda Regularization parameter"},
                      yaxis={"title": "y - MSE error"},
                      height=1000,
                      width=2000)

    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    r_lam_star: int = lam_range[np.argmin(r_validation_errors)]
    l_lam_star: int = lam_range[np.argmin(l_validation_errors)]

    ridge_fitted_model: RidgeRegression = RidgeRegression(lam=r_lam_star).fit(X_train.to_numpy(), y_train.to_numpy())
    lasso_fitted_model: Lasso = Lasso(alpha=l_lam_star).fit(X_train.to_numpy(), y_train.to_numpy())
    ls_fitted_model: LinearRegression = LinearRegression().fit(X_train.to_numpy(), y_train.to_numpy())

    ridge_test_err: float = mean_square_error(y_test.to_numpy(), ridge_fitted_model.predict(X_test.to_numpy()))
    lasso_test_err: float = mean_square_error(y_test.to_numpy(), lasso_fitted_model.predict(X_test.to_numpy()))
    ls_test_err: float = mean_square_error(y_test.to_numpy(), ls_fitted_model.predict(X_test.to_numpy()))

    print(f"Best lam reg param ro Ridge model: {np.round(r_lam_star, decimals=6)} \n"
          f"Best lam reg param ro Lasso model: {np.round(l_lam_star, decimals=6)} \n\n"
          
          f"Ridge Fitted Model Test Error: {np.round(ridge_test_err, decimals=6)} \n"
          f"Lasso Fitted Model Test Error: {np.round(lasso_test_err, decimals=6)} \n"
          f"LS Fitted Model Test Error: {np.round(ls_test_err, decimals=6)} \n")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(noise=0)
    select_polynomial_degree()
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
