import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type, Dict

from plotly.subplots import make_subplots

from IMLearn.base import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2, LogisticModule
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error
from sklearn.metrics import roc_curve, auc, accuracy_score


import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    def callback(model: GradientDescent, **kwargs):
        val_t: np.ndarray = kwargs.get('val').copy()
        weights_t: np.ndarray = kwargs.get('weights').copy()

        values.append(val_t)
        weights.append(weights_t)

    values: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # l1_fig: go.Figure = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.08)
    #
    #
    # l1_figs: List[go.Figure] = []
    # l2_figs: List[go.Figure] = []
    fig_convergence_l1: go.Figure = make_subplots(rows=2, cols=2,
                                                  subplot_titles=[rf"$\textbf{{L1 norm convergence rate with \
                                                  learning rate: eta = {eta} }}$"
                                                                  for eta in etas],
                                                  horizontal_spacing=0.08)

    fig_convergence_l2: go.Figure = make_subplots(rows=2, cols=2,
                                                  subplot_titles=[rf"$\textbf{{L2 norm convergence rate with \
                                                  learning rate: eta = {eta} }}$"
                                                                  for eta in etas],
                                                  horizontal_spacing=0.08)

    colors: List[str] = ['tomato', 'sienna', 'purple', 'olive']
    res_gd_l1: List[np.ndarray] = []
    res_gd_l2: List[np.ndarray] = []
    for i, eta in enumerate(etas):
        tmp_weights_l1, tmp_weights_l2 = init.copy(), init.copy()
        model_name_map: Dict[str, BaseModule] = {'L1': L1(weights=tmp_weights_l1),
                                                 'L2': L2(weights=tmp_weights_l2)}

        for model_name, model in model_name_map.items():
            callback, values, weights = get_gd_state_recorder_callback()
            gd: GradientDescent = GradientDescent(learning_rate=FixedLR(base_lr=eta), callback=callback)
            res_gd: np.ndarray = gd.fit(f=model, X=np.array([]), y=np.array([]))

            if model_name == 'L1':
                res_gd_l1.append(L1(weights=res_gd).compute_output())
            else:
                res_gd_l2.append(L2(weights=res_gd).compute_output())

            # Q1 - descent paths.
            descent_path: np.ndarray = np.vstack(weights)
            fig: go.Figure = plot_descent_path(module=type(model), descent_path=descent_path,
                                               title=model_name + f" eta = {eta}")
            fig.show()

            # Q3 - convergence rate
            norm_vals: np.ndarray = np.vstack(values)
            xrange = list(range(1, gd.max_iter_))

            if model_name == 'L1':
                fig_convergence_l1.add_trace(trace=go.Scatter(x=xrange, y=norm_vals.flatten(), mode="lines"
                                                              , name=f'eta = {eta}'
                                                              , marker=dict(color=colors[i], opacity=.6)
                                                              , showlegend=True), row=(i // 2) + 1, col=(i % 2) + 1)
            else:
                fig_convergence_l2.add_trace(trace=go.Scatter(x=xrange, y=norm_vals.flatten(), mode="lines"
                                                              , name=f'eta = {eta}'
                                                              , marker=dict(color=colors[i], opacity=.6)
                                                              , showlegend=True), row=(i // 2) + 1,
                                             col=(i % 2) + 1)
    fig_convergence_l1.show()
    fig_convergence_l2.show()

    # Q4
    min_eta_l1_index = np.argmin(res_gd_l1)
    min_eta_l2_index = np.argmin(res_gd_l2)

    min_eta_l1, min_value_l1 = etas[min_eta_l1_index], res_gd_l1[min_eta_l1_index]
    min_eta_l2, min_value_l2 = etas[min_eta_l2_index], res_gd_l2[min_eta_l2_index]

    print(f" For L1 model minimum eta = {min_eta_l1}, \n"
          f"With minimum value of {min_value_l1}. \n")

    print(f" For L2 model minimum eta = {min_eta_l2}, \n"
          f"With minimum value of {min_value_l2}. \n")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig_descent_path_l1: go.Figure = go.Figure()
    fig_descent_path_l2: go.Figure = go.Figure()

    fig_convergence_l1: go.Figure = make_subplots(rows=2, cols=2,
                                                  subplot_titles=[f"L1 norm convergence rate with learning "
                                                                  f"rate: eta = {eta} decay rate: gamma = {gamma}"
                                                                  for gamma in gammas],
                                                  horizontal_spacing=0.08)

    colors: List[str] = ['tomato', 'sienna', 'purple', 'olive']
    res_gd_l1: List[np.ndarray] = []
    for i, gamma in enumerate(gammas):
        tmp_weights_l1 = init.copy()
        tmp_weights_l2 = init.copy()

        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()

        gd_l1: GradientDescent = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gamma),
                                                 callback=callback_l1)

        gd_l2: GradientDescent = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gamma),
                                                 callback=callback_l2)

        l1_model: BaseModule = L1(weights=tmp_weights_l1)
        l2_model: BaseModule = L2(weights=tmp_weights_l2)

        res_gd: np.ndarray = gd_l1.fit(f=l1_model, X=np.array([]), y=np.array([]))
        gd_l2.fit(f=l2_model, X=np.array([]), y=np.array([]))

        res_gd_l1.append(L1(weights=res_gd).compute_output())

        norm_vals: np.ndarray = np.vstack(values_l1)
        xrange = list(range(1, gd_l1.max_iter_))
        fig_convergence_l1.add_trace(trace=go.Scatter(x=xrange, y=norm_vals.flatten(), mode="lines"
                                                      , name=f'eta = {eta}'
                                                      , marker=dict(color=colors[i], opacity=.6)
                                                      , showlegend=True), row=(i // 2) + 1, col=(i % 2) + 1)

        # Create fig descent path for gamma=0.95
        if gamma == .95:
            descent_path_l1: np.ndarray = np.vstack(weights_l1)
            descent_path_l2: np.ndarray = np.vstack(weights_l2)

            fig_descent_path_l1: go.Figure = plot_descent_path(module=type(l1_model), descent_path=descent_path_l1,
                                                               title='L1' + f" eta = {eta} "
                                                                            f" gamma = {.95}")

            fig_descent_path_l2: go.Figure = plot_descent_path(module=type(l2_model), descent_path=descent_path_l2,
                                                               title='L2' + f" eta = {eta} "
                                                                            f" gamma = {.95}")

    # Plot algorithm's convergence for the different values of gamma
    fig_convergence_l1.show()

    # Plot descent path for gamma=0.95
    fig_descent_path_l1.show()
    fig_descent_path_l2.show()

    min_gamma_l1_index = np.argmin(res_gd_l1)

    min_gamma_l1, min_value_l1 = gammas[min_gamma_l1_index], res_gd_l1[min_gamma_l1_index]

    print(f" For L1 model minimum gamma = {min_gamma_l1}, \n"
          f"With minimum value of {min_value_l1}. \n")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    alpha_range: List[float] = np.linspace(0, 1, 100).tolist()
    l1_model: LogisticRegression = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                                                             max_iter=20000)) \
        .fit(X_train.to_numpy(), y_train.to_numpy())

    fpr, tpr = [], []
    for alpha in alpha_range:
        l1_model.alpha_ = alpha
        y_pred: np.ndarray = l1_model.predict(X_train.to_numpy()).flatten()
        # y_train = y_train.to_numpy()
        fp = np.sum((y_pred == 1) & (y_train == 0))
        tp = np.sum((y_pred == 1) & (y_train == 1))

        fn = np.sum((y_pred == 0) & (y_train == 1))
        tn = np.sum((y_pred == 0) & (y_train == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=alpha_range, name="",
                         showlegend=False, marker_size=5,
                         marker_color='olive',
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    alpha_star: float = alpha_range[np.argmax(np.array(tpr) - np.array(fpr))]
    l1_model.alpha_ = alpha_star
    test_err: float = accuracy_score(y_test.to_numpy(), l1_model.predict(X_test.to_numpy()))
    print(f"Optimal threshold alpha_star = {alpha_star}, \n"
          f"with test error (missclassification error) = {test_err}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    lam_range: List[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    # Mean of - Train and validation errors for all values of lam
    l1_train_errors, l1_validation_errors = [], []
    l2_train_errors, l2_validation_errors = [], []
    for lam in lam_range:
        l1_model: LogisticRegression = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4)),
                                                          penalty="l1",
                                                          lam=lam)

        l2_model: LogisticRegression = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4)),
                                                          penalty="l2",
                                                          lam=lam)

        train_err, val_error = cross_validate(l1_model, X_train.to_numpy(),
                                              y_train.to_numpy().reshape([-1, 1]),
                                              misclassification_error)
        l1_train_errors.append(train_err), l1_validation_errors.append(val_error)

        train_err, val_error = cross_validate(l2_model, X_train.to_numpy(),
                                              y_train.to_numpy().reshape([-1, 1]),
                                              misclassification_error)
        l2_train_errors.append(train_err), l2_validation_errors.append(val_error)

    l1_lam_star: float = lam_range[np.argmin(l1_validation_errors)]
    l2_lam_star: float = lam_range[np.argmin(l2_validation_errors)]

    l1_fitted_model: LogisticRegression = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4)),
                                                             penalty="l1",
                                                             lam=l1_lam_star).fit(X_train.to_numpy(),
                                                                                  y_train.to_numpy())

    l2_fitted_model: LogisticRegression = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4)),
                                                             penalty="l2",
                                                             lam=l2_lam_star).fit(X_train.to_numpy(),
                                                                                  y_train.to_numpy())

    l1_test_err: float = misclassification_error(y_test.to_numpy(), l1_fitted_model.predict(X_test.to_numpy()))
    l2_test_err: float = misclassification_error(y_test.to_numpy(), l2_fitted_model.predict(X_test.to_numpy()))

    print(f"Best lam reg param ro L1 model: {np.round(l1_lam_star, decimals=6)} \n"
          f"Best lam reg param ro L2 model: {np.round(l2_lam_star, decimals=6)} \n\n"

          f"L1 Fitted Model Test Error: {np.round(l1_test_err, decimals=6)} \n"
          f"L2 Fitted Model Test Error: {np.round(l2_test_err, decimals=6)} \n")

    # print(f"Regularization Parameter {lam} \n"
    #       f"Average Train Error: {train_err} \n"
    #       f"Average Validation Error: {val_error} \n\n")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
