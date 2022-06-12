import numpy as np
from IMLearn.base import BaseEstimator
from typing import Callable, NoReturn, List
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner
    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator
    self.iterations_: int
        Number of boosting iterations to perform
    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator
        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator
        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Set initial distribution over X - uniform
        self.fitted_ = True

        m: int = X.shape[0]
        D_t: np.ndarray = np.full(len(X), fill_value=1 / float(m))

        # Init memory of adaboost.
        self.models_: List[BaseEstimator] = []
        self.weights_: List[float] = []

        for t in range(self.iterations_):
            print(t)
            # Get fitted model and prediction for iteration t.
            model_t: BaseEstimator = self.wl_()
            pred_t: np.ndarray = model_t.fit_predict(X, y * D_t)

            # Calculate weight for iteration t.
            epsilon_t: float = np.sum(np.where(pred_t != y, 1, 0) * D_t)
            w_t: float = 0.5 * np.log((1 / epsilon_t) - 1)

            # Append model_t, w_t, D_t, to adaboost memory.
            self.models_.append(model_t)
            self.weights_.append(w_t)

            # Update D_t to be D_t+1, and normalize D_t+1
            D_t *= np.exp(-y * w_t * pred_t)
            normalize_factor: float = np.sum(D_t)
            D_t /= normalize_factor

        self.D_ = D_t

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `predict` function")

        y_pred: np.ndarray = np.zeros(X.shape[0])
        for t in range(T):
            model_t: BaseEstimator = self.models_[t]
            weight_t: float = self.weights_[t]
            y_pred += weight_t * model_t.predict(X)

        return np.sign(y_pred)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return misclassification_error(np.sign(y), np.sign(self.partial_predict(X, T)))
