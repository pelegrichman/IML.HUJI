from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(model: GradientDescent, **kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm
    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm
    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance
    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training
    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations
    callback_: Callable[[GradientDescent, ...], None]
        A callable function to be called after each update of the model while fitting to given data
        Callable function should receive as input a GradientDescent instance, and any additional
        arguments specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class
        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm
        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance
        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training
        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes
        callback: Callable[[GradientDescent, ...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data
            Callable function should receive as input a GradientDescent instance, and any additional
            arguments specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses
        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over
        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_
        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_
        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step
        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)
        """

        # Init first weights vector and delta value.
        weights_t_m1: np.ndarray = f.weights
        delta_t: float = np.infty

        # Init min val weights vector, and min val reached.
        weights_t_star: np.ndarray = weights_t_m1
        val_star: np.ndarray = f.compute_output(X=X, y=y)

        # Init Max iter the algo reached, sum of weights_t
        # for avg solution.
        T: int = 1
        weights_t_sum: np.ndarray = weights_t_m1

        self.callback_(self, **dict(weights=weights_t_m1,
                                    val=val_star))

        for t in range(1, self.max_iter_ + 1):
            if delta_t <= self.tol_:
                break

            # Update eta_t and weights_t and delta_t
            eta: float = self.learning_rate_.lr_step(t=t)
            grad: np.ndarray = f.compute_jacobian(X=X, y=y)
            weights_t: np.ndarray = weights_t_m1 - (eta * grad)
            delta_t = np.linalg.norm(x=grad)

            # Set weights
            f.weights = weights_t
            weights_t_m1 = weights_t

            # Update t_star if needed
            val: np.ndarray = f.compute_output(X=X, y=y)
            if np.less(val, val_star).all():
                val_star = val
                weights_t_star = weights_t

            # Update weights sum, and T
            weights_t_sum += weights_t
            T = t

            self.callback_(self, **dict(weights=weights_t,
                                        val=val, grad=grad,
                                        t=t, eta=eta,
                                        delta=delta_t))

        # Last weights vector w_T
        if self.out_type_ == OUTPUT_VECTOR_TYPE[0]:
            return f.weights
        # Best weights vector - lowest val.
        elif self.out_type_ == OUTPUT_VECTOR_TYPE[1]:
            return weights_t_star
        # Avg weights vector
        else:
            return (1 / T) * weights_t_sum
