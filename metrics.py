import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = y_pred[(y_true == y_pred) & (y_pred == 1)].shape[0]  # true positive
    fp = y_pred[(y_true != y_pred) & (y_pred == 1)].shape[0]  # false positive
    tn = y_pred[(y_true == y_pred) & (y_pred == 0)].shape[0]  # true negative
    fn = y_pred[(y_true != y_pred) & (y_pred == 0)].shape[0]  # false negative
    
    if tp + tn == 0:
        answer = {'presicion': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        return answer
    elif fp + fn == 0:
        answer = {'presicion': 1, 'recall': 1, 'f1': 1, 'accuracy': 1}
        return answer
    
    pr_score = tp / (tp + fp) # precision = TP / (TP + FP)
    rec_score = tp / (tp + fn) # recall = TP / (TP + FN)
    
    if tp == 0:
        f1_score = 0
    else:
        f1_score = (2 * pr_score * rec_score) / (pr_score + rec_score) # f1 = (2 * Pr * Rec) / (Pr + Rec)
    
    acc_score = (tp + tn) / len(y_pred) # accuracy = (TP + TN) / (P + N) or (TP + TN) / len(y_pred)
    answer = {'presicion': pr_score, 'recall': rec_score, 'f1': f1_score, 'accuracy': acc_score}    
    return answer


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    t = y_pred[(y_true == y_pred)].shape[0]
    acc_score = t / y_true.shape[0]
    return acc_score


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    sum_pred = np.sum((y_true - y_pred)**2)
    sum_mean = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (sum_pred / sum_mean)


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    sum_pred = np.sum((y_true - y_pred)**2)
    return sum_pred / y_pred.shape[0]


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    sum_pred = np.sum(np.abs(y_true - y_pred))
    return sum_pred / y_pred.shape[0]
    