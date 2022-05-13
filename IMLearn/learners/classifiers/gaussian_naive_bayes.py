from typing import NoReturn, Any, Dict
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import inv, det
from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None
        self.label_mu_index_map_: Dict[Any, int] = {}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_: np.ndarray = np.asarray(list(set(y)))
        K: int = len(self.classes_)
        n: int = X.shape[1]
        m: int = X.shape[0]
        self.mu_: np.ndarray = np.zeros([K, n])
        self.pi_: np.ndarray = np.zeros([K, 1])
        self.vars_: np.ndarray = np.zeros([K, n])

        for i, k in enumerate(self.classes_):
            # Sum of indicators for how many labels
            # Are in a certain class - k
            n_k: int = sum([int(label == k) for label in y])
            sum_of_x_y_i_in_k: np.ndarray = np.zeros([1, n])
            for label, sample in (zip(y, X)):
                if label == k:
                    sum_of_x_y_i_in_k += sample

            self.mu_[i] = sum_of_x_y_i_in_k / n_k
            self.pi_[i] = n_k / m
            self.label_mu_index_map_[k] = i

            sum_of_cov = np.zeros([n, n])
            for label, sample in (zip(y, X)):
                if label == k:
                    sample_minus_mu_given_label: np.ndarray = sample - self.mu_[self.label_mu_index_map_[label]]
                    sum_of_cov += np.reshape(sample_minus_mu_given_label, (-1, 1)) \
                                  @ np.reshape(sample_minus_mu_given_label, (1, -1))

            self.vars_[i] = np.diag(sum_of_cov / n_k)

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

                cov_k: np.ndarray = np.zeros([d, d])
                np.fill_diagonal(cov_k, self.vars_[self.label_mu_index_map_[class_j]])
                cov_inv_k: np.ndarray = inv(cov_k)

                gau_norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det(cov_k))
                gau_norm_exp = np.exp(-0.5 * ((sample - mu_k).reshape((1, -1))) @ cov_inv_k @ (sample - mu_k))
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
