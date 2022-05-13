from typing import NoReturn, Any, Dict
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None
        self.label_mu_index_map_: Dict[Any, int] = {}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # init classes, K = num of classes, n = num of features.
        self.classes_: np.ndarray = np.asarray(list(set(y)))
        K: int = len(self.classes_)
        n: int = X.shape[1]
        m: int = X.shape[0]
        self.mu_: np.ndarray = np.zeros([K, n])
        self.pi_: np.ndarray = np.zeros([K, 1])

        for i, k in enumerate(self.classes_):
            # Sum of indicators for how many labels
            # Are in a certain class - k
            n_k: int = 0
            sum_of_x_y_i_in_k: np.ndarray = np.zeros([1, n])
            for label, sample in (zip(y, X)):
                if label == k:
                    sum_of_x_y_i_in_k += sample
                    n_k += 1

            self.mu_[i] = sum_of_x_y_i_in_k / n_k
            self.pi_[i] = n_k / m
            self.label_mu_index_map_[k] = i

        sum_of_cov = np.zeros([n, n])
        for label, sample in (zip(y, X)):
            sample_minus_mu_given_label: np.ndarray = sample - self.mu_[self.label_mu_index_map_[label]]
            sum_of_cov += np.reshape(sample_minus_mu_given_label, (-1, 1)) \
                          @ np.reshape(sample_minus_mu_given_label, (1, -1))

        self.cov_: np.ndarray = sum_of_cov / m
        self._cov_inv: np.ndarray = inv(self.cov_)
        self.fitted_: bool = True

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
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `predict` function")

        # prediction = for every sample, take argmax on sample's likelihood for each class,
        # And return pred sample i, the maximizer likelihood class.
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        d: int = X.shape[1]
        likelihoods: np.ndarray = np.zeros([len(X), len(self.classes_)])

        for i, sample in enumerate(X):
            sample: np.ndarray = sample.reshape((-1, 1))
            for j, class_j in enumerate(self.classes_):
                mult_k: float = self.pi_[self.label_mu_index_map_[class_j]]
                mu_k: np.ndarray = self.mu_[self.label_mu_index_map_[class_j]].reshape((-1, 1))

                gau_norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det(self.cov_))
                gau_norm_exp = np.exp(-0.5 * ((sample - mu_k).reshape((1, -1))) @ self._cov_inv @ (sample - mu_k))
                normal_x_given_y = gau_norm_factor * gau_norm_exp

                likelihoods[i, j] = mult_k * normal_x_given_y
        return likelihoods

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
        return misclassification_error(y, self.predict(X))
