from __future__ import annotations
from typing import Tuple, NoReturn, List
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm
    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split
    self.j_ : int
        The index of the feature by which to split the data
    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self):
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        signs: np.ndarray = np.asarray([1, -1])
        feature_num: int = X.shape[1]
        self.j_: int = 0
        self.sign_: int = 0
        self.threshold_: float = 0

        min_err: float = np.inf
        for sign in signs:
            for coordinate in range(feature_num):
                opt_thr, thr_err = self._find_threshold(X[:, coordinate], y, sign)
                if thr_err < min_err:
                    min_err = thr_err
                    self.threshold_ = opt_thr
                    self.j_ = coordinate
                    self.sign_ = sign
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `predict` function")

        y_pred: np.ndarray = np.zeros(len(X))
        for i in range(len(X)):
            y_pred[i]: int = self.sign_ if X[i][self.j_] >= self.threshold_ else -self.sign_

        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature
        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for
        labels: ndarray of shape (n_samples,)
            The labels to compare against
        sign: int
            Predicted label assigned to values equal to or above threshold
        Returns
        -------
        thr: float
            Threshold by which to perform split
        thr_err: float between 0 and 1
            Misclassification error of returned threshold
        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        opt_thr: float = 0
        min_thr_err: float = np.inf
        for thr in values:
            labels_pred: np.ndarray = np.where(values >= thr, sign, -sign)

            err_indices = np.sign(labels) != np.sign(labels_pred)
            thr_err = np.sum(np.absolute(labels[err_indices]))

            if thr_err < min_thr_err:
                min_thr_err = thr_err
                opt_thr = thr

        return opt_thr, min_thr_err

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
            Performance under misclassification loss function
        """

        y_pred = self.predict(X)
        err_indices = np.sign(y) != np.sign(y_pred)
        return np.sum(np.absolute(y[err_indices]))
