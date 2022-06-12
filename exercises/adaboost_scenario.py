import numpy as np
from typing import Tuple, List, Callable
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers.decision_stump import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners: int = 250, train_size: int = 5000, test_size: int = 500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_model: AdaBoost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    ada_model.fit(train_X, train_y)
    train_losses: List[float] = []
    test_losses: List[float] = []

    for i in range(1, n_learners + 1):
        loss: float = ada_model.partial_loss(train_X, train_y, T=i)
        train_losses.append(loss)

        loss: float = ada_model.partial_loss(test_X, test_y, T=i)
        test_losses.append(loss)

    fig = go.Figure([go.Scatter(x=list(range(1, n_learners)), y=train_losses, name='Error of Train',
                                mode="lines", line=dict(color="darkblue")),
                     go.Scatter(x=list(range(1, n_learners)), y=test_losses, name='Error of Test',
                                mode="lines", line=dict(color="darkred"))],
                    layout=go.Layout(title=f"<b>Adaboost using Decision Stump misclassification error, "
                                           f"as a function of fitted learners "
                                           f"Noise is {noise}.</b>",
                                     xaxis=dict(title="<b>number of fitted learners </b>"),
                                     yaxis=dict(title="<b>misclassification error</b>")))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"<b>Ensemble up to iteration : {t}<b>" for t in T],
                        horizontal_spacing=0.08)

    color = test_y.astype(dtype=int)
    for i in range(len(T)):
        num_of_learners: int = T[i]
        fig.add_traces([decision_surface(lambda X: ada_model.partial_predict(X, num_of_learners), lims[0], lims[1],
                                         showscale=False,
                                         colorscale=class_colors(2), density=500),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=color,
                                               symbol=class_symbols[color], colorscale=class_colors(2),
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        title=rf"$\textbf{{(2) Decision Boundaries for different number of fitted learners, Noise is {noise}}}$",
        margin=dict(t=100), yaxis1_range=[-1, 1], yaxis2_range=[-1, 1],
        yaxis3_range=[-1, 1], yaxis4_range=[-1, 1],
        xaxis1_range=[-1, 1], xaxis2_range=[-1, 1],
        xaxis3_range=[-1, 1], xaxis4_range=[-1, 1]) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()

    # Question 3: Decision surface of best performing ensemble
    opt_num_of_learners: int = np.argmin(test_losses)
    opt_accuracy = accuracy(test_y, ada_model.partial_predict(test_X, opt_num_of_learners))
    fig = go.Figure([decision_surface(lambda X: ada_model.partial_predict(X, opt_num_of_learners), lims[0], lims[1],
                                      showscale=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                mode='markers', showlegend=False,
                                marker=dict(color=color, symbol=class_symbols[color], colorscale=class_colors(2),
                                            line=dict(color="black", width=1)))])

    fig.update_layout(height=800, width=1100,
                      title=f'<b> Lowest test error was achieved by ensemble with size: {opt_num_of_learners}</b><br>'
                            f'<b> with accuracy equals to: {opt_accuracy}, Noise is {noise}</b><br>') \
        .update_xaxes(range=[-1, 1], visible=False) \
        .update_yaxes(range=[-1, 1], visible=False) \
        .show()

    # Question 4: Decision surface with weighted samples
    D_T = ada_model.D_ / np.max(ada_model.D_) * 5
    fig = go.Figure([decision_surface(lambda X: ada_model.predict(X), lims[0], lims[1], showscale=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                mode='markers', showlegend=False,
                                marker=dict(color=color, size=D_T, symbol=class_symbols[color],
                                            colorscale=class_colors(2), line=dict(color="black", width=1)))])

    fig.update_layout(height=500, width=800,
                      title=f'<b>training set with a point size proportional to it’s </b><br>'
                            f'<b>weight and color (and/or shape) indicating its label </b><br>'
                            f'><b>using the weights of the last iteration (full ensemble)'
                            f' , Noise is {noise}</b><br>') \
        .update_xaxes(range=[-1, 1], visible=False) \
        .update_yaxes(range=[-1, 1], visible=False) \
        .show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
