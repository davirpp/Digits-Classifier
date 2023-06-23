import numpy
import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    """
    Computes a confusion matrix from two arrays.

    Parameters
    ----------
    y_true : numpy.ndarray
        The real values of the labels
    y_pred : numpy.ndarray
        Predicted labels 
    labels : list
        A list of all possible labels.

    Returns
    -------
    numpy.ndarray
        The confusion matrix
    """

    n = len(labels)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    matriz = np.zeros((n, n), dtype=int)

    for id, i in enumerate(labels):
        for jd, j in enumerate(labels):
            matriz[id][jd] = np.sum((y_true == i) & (y_pred == j))

    return matriz

def precision_score(y_true: numpy.ndarray, y_pred: numpy.ndarray, label: int) -> float:
    """
    Computes the precision score for a classification model.

    Parameters
    ----------
    y_true : numpy.ndarray
        The real values of the labels
    y_pred : numpy.ndarray
        Predicted labels
    label : int
        The label (0, 1, 4, 5) for which to compute the precision score.

    Returns
    -------
    float
        The precision score, defined as the proportion of 
        true positive(TP) predictions among all 
        predicted positive (TP + FP) examples.
    """
    return (np.sum((y_true == label) & (y_pred == label)) / np.sum(y_pred == label))

def recall_score(y_true: numpy.ndarray, y_pred: numpy.ndarray, label: int) -> float:
    """
    Computes the recall score for a classification model.

    Parameters
    ----------
    y_true : numpy.ndarray
        The real values of the labels
    y_pred : numpy.ndarray
        Predicted labels
    label : int
        The label (0, 1, 4, 5) for which to compute the precision score.

    Returns
    -------
    float
        The recall score, defined as the proportion of 
        true positive(TP) predictions among all true positive(TP + FN) 
        examples.
    """
    return np.sum((y_true == label) & (y_pred == label)) / np.sum(y_true == label)

def f1_score(y_true: numpy.ndarray, y_pred: numpy.ndarray, label: int) -> float:
    """
    Computes the F1 score for a classification model.

    Parameters
    ----------
    y_true : numpy.ndarray
        The real values of the labels
    y_pred : numpy.ndarray
        Predicted labels
    label : int
        The label (0, 1, 4, 5) for which to compute the precision score.

    Returns
    -------
    float
        The F1 score, defined as the harmonic mean of precision and 
        recall for the specified label.
    """
    precision = precision_score(y_true, y_pred, label)
    recall = recall_score(y_true, y_pred, label)
    return 2 * precision * recall / (precision + recall)

def accuracy_score(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
    """
    Computes the accuracy score for a classification model.

    Parameters
    ----------
    y_true : numpy.ndarray
        The real values of the labels
    y_pred : numpy.ndarray
        Predicted labels

    Returns
    -------
    float
        The accuracy score, defined as the proportion of correct 
        predictions.

    """
    return np.sum(y_true == y_pred) / len(y_true)

def classification_report(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> str:
    """
    Builds a text report showing the main classification metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        The real values of the labels
    y_pred : numpy.ndarray
        Predicted labels

    Returns
    -------
    str
        A string representation of the classification report.
    """
    top = f"{' '*4}precision    recall  f1-score   support\n"
    labels_info = []
    labels = list(np.unique(y_true))
    for i in range(len(labels)):
        labels_info.append(f"{' '}{labels[i]}{' '*6} "
                           f"{precision_score(y_true, y_pred, labels[i]):.2f}{' '*5} "
                           f"{recall_score(y_true, y_pred, labels[i]):.2f}{' '*5} "
                           f"{f1_score(y_true, y_pred, labels[i]):.2f}{' '*6} "
                           f"{np.sum(y_true == labels[i])}")
    accuracy = accuracy_score(y_true, y_pred)
    total = len(y_pred)
        
    return top + '\n' + '\n'.join(labels_info) + '\n\n' + f"{'accuracy':>8} {accuracy:.2f} {total:>29}"
