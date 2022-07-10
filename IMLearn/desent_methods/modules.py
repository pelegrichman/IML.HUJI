import numpy as np
from IMLearn import BaseModule
from typing import Type


class L2(BaseModule):
    """
    Class representing the L2 module

    Represents the function: f(w)=||w||^2_2
    """

    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L2 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """

        return np.linalg.norm(self.weights)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L2 derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L2 derivative with respect to self.weights at point self.weights
        """
        # if self.weights.all() == 0:
        #     return np.zeros(shape=self.shape)
        return 2 * self.weights


class L1(BaseModule):
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L1 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        return np.linalg.norm(self.weights, 1)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L1 derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L1 derivative with respect to self.weights at point self.weights
        """
        if self.weights.all() == 0:
            return np.ones(shape=self.shape)
        return np.sign(self.weights)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


class LogisticModule(BaseModule):
    """
    Class representing the logistic regression objective function

    Represents the function:  f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))]
    """

    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a logistic regression module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the output value of the logistic regression objective function at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        # Represents the function:  f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))]
        m: int = len(X)
        x_w: np.ndarray = np.dot(X, self.weights)
        return -(1 / m) * np.sum((y * x_w) - np.log(1 + np.exp(x_w)))

        # return -(1 / m) * np.sum([((y * np.inner(x_i, self.weights)) -
        #                            (np.log(1 + np.exp(np.inner(x_i, self.weights))))) for x_i in X])

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the logistic regression objective function at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (n_features,)
            Derivative of function with respect to self.weights at point self.weights
        """
        # dot(X.T, self.sigmoid(dot(X, params)) - np.reshape(y, (len(y), 1)))

        m: int = len(X)
        x_w: np.ndarray = np.dot(X, np.reshape(self.weights, (len(self.weights), 1)))
        return 1 / m * np.dot(X.T, sigmoid(x_w) - np.reshape(y, (len(y), 1)))


class RegularizedModule(BaseModule):
    """
    Class representing a general regularized objective function of the format:
                                    f(w) = F(w) + lambda*R(w)
    for F(w) being some fidelity function, R(w) some regularization function and lambda
    the regularization parameter
    """

    def __init__(self,
                 fidelity_module: BaseModule,
                 regularization_module: BaseModule,
                 lam: float = 1.,
                 weights: np.ndarray = None,
                 include_intercept: bool = True):
        """
        Initialize a regularized objective module instance

        Parameters:
        -----------
        fidelity_module: BaseModule
            Module to be used as a fidelity term

        regularization_module: BaseModule
            Module to be used as a regularization term

        lam: float, default=1
            Value of regularization parameter

        weights: np.ndarray, default=None
            Initial value of weights

        include_intercept: bool default=True
            Should fidelity term (and not regularization term) include an intercept or not
        """
        super().__init__()
        self.fidelity_module_, self.regularization_module_, self.lam_ = fidelity_module, regularization_module, lam
        self.include_intercept_ = include_intercept

        if weights is not None:
            if self.include_intercept_:
                weights = np.vstack((np.ones(shape=(1,)), weights))
            self.weights = weights

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the regularized objective function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        X_f: np.ndarray = kwargs.get('X', None)
        weights_f: np.ndarray = self.weights
        y: np.ndarray = kwargs.get('y', None)

        if self.include_intercept_:
            X_f = np.hstack((np.ones(shape=(X_f.shape[0],1)), X_f))
            X_r: np.ndarray = X_f[:, 1:]
            weights_r: np.ndarray = weights_f[1:]
        else:
            X_r = X_f
            weights_r = weights_f

        f_model: Type[BaseModule] = type(self.fidelity_module_)
        out_fidelity: np.ndarray = f_model(weights=weights_f).compute_output(X=X_f, y=y)
        if self.regularization_module_ is None:
            return out_fidelity

        r_model: Type[BaseModule] = type(self.regularization_module_)
        out_regularization: np.ndarray = r_model(weights=weights_r).compute_output()
        return out_fidelity + (self.lam_ * out_regularization)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            Derivative with respect to self.weights at point self.weights
        """
        X_f: np.ndarray = kwargs.get('X', None)
        weights_f: np.ndarray = self.weights
        y: np.ndarray = kwargs.get('y', None)

        if self.include_intercept_:
            X_f = np.hstack((np.ones(shape=(X_f.shape[0], 1)), X_f))
            X_r: np.ndarray = X_f[:, 1:]
            weights_r: np.ndarray = weights_f[1:]
        else:
            X_r = X_f
            weights_r = weights_f

        f_model: Type[BaseModule] = type(self.fidelity_module_)
        out_fidelity: np.ndarray = f_model(weights=weights_f).compute_jacobian(X=X_f, y=y)
        if self.regularization_module_ is None:
            return out_fidelity

        r_model: Type[BaseModule] = type(self.regularization_module_)
        out_regularization: np.ndarray = r_model(weights=weights_r).compute_jacobian()

        if self.include_intercept_:
            out_regularization = np.vstack((np.ones(shape=(1,)), out_regularization))
        return out_fidelity + (self.lam_ * out_regularization)

    @property
    def weights(self):
        """
        Wrapper property to retrieve module parameter

        Returns
        -------
        weights: ndarray of shape (n_in, n_out)
        """
        return self.weights_

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """
        Setter function for module parameters

        In case self.include_intercept_ is set to True, weights[0] is regarded as the intercept
        and is not passed to the regularization module

        Parameters
        ----------
        weights: ndarray of shape (n_in, n_out)
            Weights to set for module
        """
        self.weights_ = weights
