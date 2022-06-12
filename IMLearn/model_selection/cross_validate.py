from __future__ import annotations
from typing import Tuple, Callable, List, Dict
import numpy as np

from IMLearn import BaseEstimator
from itertools import combinations

# Types
Fold = Tuple[np.ndarray, np.ndarray]
DS_Partition = List[Fold]
Training_Sets = List[Tuple[Fold]]


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # Step 1: Divide training set to cv training sets - where in each set,
    # there are cv-1 folds of the data, and the missing fold is the validation set.
    fold_size: int = len(X) // cv
    split_indices: np.array = np.arange(start=fold_size, stop=len(X) - fold_size + 1, step=fold_size)
    dataset_cv_partition: Dict[int, Fold] = {i: fold for i, fold in
                                             enumerate(zip(np.split(X, split_indices),
                                                           np.split(y, split_indices)))}

    folds_indices: List[int] = list(dataset_cv_partition.keys())
    training_sets_by_fold_index: List[Tuple[int]] = list(combinations(folds_indices, r=cv - 1))

    training_errors: np.ndarray = np.zeros(cv, dtype=float)
    validation_errors: np.ndarray = np.zeros(cv, dtype=float)
    for training_set in training_sets_by_fold_index:

        valid_index: int = list(set(folds_indices) - set(training_set))[0]
        folds_X: List[np.ndarray] = []
        folds_y: List[np.ndarray] = []
        for fold_index in training_set:
            folds_X.append(dataset_cv_partition[fold_index][0])
            folds_y.append(dataset_cv_partition[fold_index][1])

        train_X, train_y = np.vstack(folds_X), \
                           np.vstack(folds_y)

        # Fit estimator for each train set (k-1 stacked folds),
        # and calculate error over validation fold (The k-th fold that was set aside).
        validation_fold: Fold = dataset_cv_partition[valid_index]
        estimator.fit(train_X, train_y)
        pred_train_y: np.ndarray = estimator.predict(train_X)
        pred_validation_y: np.ndarray = estimator.predict(validation_fold[0])
        training_errors[valid_index] = scoring(train_y.ravel(), pred_train_y.ravel())
        validation_errors[valid_index] = scoring(validation_fold[1].ravel(), pred_validation_y.ravel())

    return np.mean(training_errors), np.mean(validation_errors)
