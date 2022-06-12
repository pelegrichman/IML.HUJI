import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return (1 / y_true.shape[0]) * (np.sum((y_true - y_pred) ** 2))


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    loss: float = 0
    for i in range(y_true.shape[0]):
        if y_true[i] * y_pred[i] <= 0:
            loss += 1

    if normalize:
        return loss / len(y_true)
    return loss


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    # Return true positive + true negative / (total positive + total negative)



    # for t_label, pred_label in zip(y_true, y_pred):
    #     # true label = positive
    #     if t_label == pred_label:
    #         if pred_label > 0:
    #             tp += 1
    #     # true label = negative
    #     else:
    #         if pred_label <= 0:
    #             tn += 1

    # res: float = (tp + tn) / len(y_true)
    return np.sum([int(l_p == l_t) for l_p, l_t in zip(y_pred, y_true)]) / len(y_true)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
