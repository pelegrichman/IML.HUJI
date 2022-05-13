from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
import os
from math import atan2, pi
from typing import Tuple, Union, List
from utils import *

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    file_path: str = os.path.join(r'..\datasets', filename)
    dataset: np.ndarray = np.load(file_path)
    return dataset[:, :-1], dataset[:, -1]


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        file_path: str = os.path.join(r'..\datasets', f)
        dataset: np.ndarray = np.load(file_path)
        X, y = dataset[:, :-1], dataset[:, -1]
        losses = []

        """ Fit Perceptron and record loss in each fit iteration
         Callback func that record the loss, of the semi-fitted 
         perceptron in each iteration of the fitting procedure 
         The loss is calculated with the miss-classification error func.
         The calculated loss depends on data and current coefs of the model. """

        def callback_training_loss(fit: Perceptron, curr_sample: np.ndarray, curr_label: int):
            X_1 = X
            if fit.include_intercept_:
                X_1 = np.c_[np.ones(len(X)), X]

            loss_i: float = fit.loss(X_1, y)
            fit.training_loss_.append(loss_i)
            losses.append(loss_i)

        perceptron: Perceptron = Perceptron(callback=callback_training_loss)
        perceptron.fit(X, y)

        # Plot figure
        fig = go.Figure([go.Scatter(x=list(range(len(losses))), y=losses, fill=None,
                                    mode="lines", line=dict(color="darkblue"))],
                        layout=go.Layout(title=f"Perceptron algorithm's training loss values (y-axis), <br>"
                                               f" as a function of the training iterations (x-axis).<br>"
                                               f"With a {n} Dataset",
                                         xaxis=dict(title="training iterations"),
                                         yaxis=dict(title="training loss values"),
                                         showlegend=False))
        fig.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:

        # Load dataset
        X, y = load_dataset(f)
        lims: np.ndarray = np.array([X.min(axis=0), X.max(axis=0)])

        # Fit models and predict over training set
        models_names: List[str] = ['GNB', 'LDA']
        fitted_models: List[Union[LDA, GaussianNaiveBayes]] = [GaussianNaiveBayes().fit(X, y),
                                                               LDA().fit(X, y)]
        predictions: List[np.ndarray] = [fitted_models[0].predict(X),
                                         fitted_models[1].predict(X)]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[rf"$\textbf{{{m}  accuracy = {np.round(accuracy(y, p), 3)}}}$"
                                            for m, p in zip(models_names, predictions)],
                            horizontal_spacing=0.08)

        for i, (name, model) in enumerate(zip(models_names, fitted_models)):

            color = y.astype(dtype=int)
            fig.add_traces([decision_surface(model.predict, lims[0], lims[1], showscale=False,
                                             colorscale=class_colors(3), density=500),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=color,
                                                   symbol=class_symbols[color], colorscale=class_colors(3),
                                                   line=dict(color="black", width=1)))],
                           rows=(i // 3) + 1, cols=(i % 3) + 1)

            for class_k in model.classes_:
                if name == 'LDA':
                    cov = model.cov_
                else:
                    cov: np.ndarray = np.zeros([X.shape[1], X.shape[1]])
                    np.fill_diagonal(cov, model.vars_[model.label_mu_index_map_[class_k]])

                mu = model.mu_[model.label_mu_index_map_[class_k]]
                fig.add_traces(get_ellipse(mu, cov),
                               rows=(i // 3) + 1, cols=(i % 3) + 1)

                fig.add_traces(go.Scatter(x=[mu[0]],
                                         y=[mu[1]],
                                         mode="markers", showlegend=False,
                                         marker=dict(color='black', symbol='x')),
                              rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - {f} Dataset}}$",
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
