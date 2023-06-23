import numpy
import numpy as np
import random
from numpy import linalg as LA
from tqdm import tqdm


def buildPCI(X: numpy.ndarray, y: numpy.ndarray, w: numpy.ndarray) -> tuple:
    """
    Builds a set of misclassified points (PCI) based on the current 
    weight vector.

    Parameters
    ----------
    X : numpy.ndarray
        A matrix containing the input data.
    y : numpy.ndarray
        A vector containing the target labels (+1 or -1).
    w : numpy.ndarray
        A vector containing the weights of the linear classifier.

    Returns
    -------
    Tuple(numpy.ndarray, numpy.ndarray)
        A tuple containing the set of misclassified points and 
        corresponding labels.
    """
    
    h = np.sign(X.dot(w))
    bool_index = (h != y)
    PCI = X[bool_index]
    Y = y[bool_index]

    return np.array(PCI), np.array(Y)


class PocketPLA():
    """
    A class that implements the Pocket Perceptron Learning Algorithm 
    (PLA) for binary classification.

    Attributes
    ----------
    w : numpy.ndarray
        The weight vector of the classifier.
    n_iter : int
        The number of iterations to run the algorithm.
    """

    def __init__(self, n_iter):
        self.w = None
        self.n_iter = n_iter
    
    def set_w(self, w: numpy.ndarray):
        """
        Sets the weight vector.

        Parameters
        ----------
        w : numpy.ndarray
            The weight vector.
        """
        self.w = w

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        """
        Fits the classifier to the given data and labels using the 
        pocket PLA algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            An array of  that contains the data points.
        y : numpy.ndarray
            An array that contains the labels (-1 or 1) for each data 
            point.
        """

        PCI = X.copy()
        orig_y = y.copy()
        if self.w is None:
            self.w = np.zeros(X.shape[1])
        best_w = self.w
        best_error = len(y)
        for _ in tqdm(range(self.n_iter)):
            if len(PCI) == 0:
                break
            rand_index = np.random.randint(len(PCI))
            x = PCI[rand_index]

            self.w += y[rand_index] * x

            PCI, y = buildPCI(X, orig_y, self.w)
            error = len(PCI)
            if error < best_error:
                best_error = error
                best_w = self.w
        self.w = best_w
    
    def predict(self, X: numpy.ndarray):
        """
        Predicts the labels for the given data using the weight vector.

        Parameters
        ----------
        X : numpy.ndarray
            An array that contains the data points.

        Returns
        -------
        numpy.ndarray
            An array that contains the predicted labels (-1 or 1) for 
            each data point.
        """
        return np.sign(X.dot(self.w))
    
    def get_w(self):
        """
        Returns the weight vector.

        Returns
        ----------
        w : numpy.ndarray
            The weight vector.
        """
        return self.w
    

class LinearRegression:
    """
    A class that implements the linear regression algorithm for binary 
    classification.

    Attributes
    ----------
    w : numpy.ndarray
        The weight vector of the classifier.
    """

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        """
        Fits the classifier to the given data and labels using the 
        pseudo inverse of X

        Parameters
        ----------
        X : numpy.ndarray
            An array that contains the data points.
        y : numpy.ndarray
            An array  that contains the labels (-1 or 1) for each data 
            point.

        Returns
        -------
        w : numpy.ndarray
            The weight vector of the classifier.
        """
        XTX_inv = np.linalg.inv(X.T.dot(X))
        pseudo_inv = XTX_inv.dot(X.T)
        self.w = pseudo_inv.dot(y)
        return self.w
     
    def predict(self, X: numpy.ndarray):
        """
        Predicts the label for a given data point using the weight vector.

        Parameters
        ----------
        X : numpy.ndarray
            An array that contains the data point.

        Returns
        -------
        int
            The predicted label (-1 or 1) for the data point.
        """
        x = np.array(X)
        return np.sign(x.dot(self.w))
     
    def get_w(self):
        """
        Returns the weight vector.

        Returns
        ----------
        w : numpy.ndarray
            The weight vector.
        """
        return self.w
    
    def set_w(self, w: numpy.ndarray):
        """
        Sets the weight vector.

        Parameters
        ----------
        w : numpy.ndarray
            The weight vector.
        """
        self.w = w


class LogisticRegression:
    """
    A class that implements the logistic regression algorithm for binary 
    classification.

    Parameters
    ----------
    eta : float, optional
        The learning rate for the gradient descent algorithm 
        (default is 0.1).
    tmax : int, optional
        The maximum number of iterations for the gradient descent 
        algorithm (default is 1000).
    batch_size : int, optional
        The size of the mini-batch for the stochastic gradient descent 
        algorithm (default is 1000000).

    Attributes
    ----------
    eta : float
        The learning rate for the gradient descent algorithm.
    tmax : int
        The maximum number of iterations for the gradient descent 
        algorithm.
    batch_size : int
        The size of the mini-batch for the stochastic gradient descent 
        algorithm.
    w : numpy.ndarray
        The weight vector of the classifier.
    """
    def __init__(self, eta=0.1, tmax=1000, batch_size=1000000, validation_lambda=0):
        self.eta = eta
        self.tmax = tmax
        self.batch_size = batch_size
        self.validation_lambda = validation_lambda

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        """
        Fits the classifier to the given data and labels using the 
        stochastic gradient descent algorithm depending on the size of 
        batch.

        Parameters
        ----------
        X : numpy.ndarray
            An array that contains the data points.
        y : numpy.ndarray
            An array that contains the labels (-1 or 1) for each data 
            point.
        """

        N = X.shape[0]
        d = X.shape[1]

        w = np.zeros(d)

        for _ in tqdm(range(int(self.tmax))):

            if self.batch_size < N:
                indexes = random.sample(range(N), self.batch_size)
                X_ = X[indexes]
                y_ = y[indexes]
                N_ = self.batch_size
            else:
                X_ = X
                y_ = y
                N_ = N

            yhat = (w @ X_.T).reshape(-1, 1)
            y_ = y_.reshape(-1, 1)

            grad = (-1/N_ * np.sum((y_ * X_) / (1 + np.exp(y_ * yhat)), axis=0)) + (2 * self.validation_lambda * w)

            if LA.norm(grad) < 1e-6:
                break
            w -= self.eta*grad

        self.w = w
    
    def predict_prob(self, X: numpy.ndarray):
        """
        Predicts the probability of the class for the given data using 
        the weight vector.

        Parameters
        ----------
        X : numpy.ndarray
            An array that contains the data points.
            
        Returns
        -------
        numpy.ndarray
            An array  that contains the predicted probabilities for each
            data point.
        """
        return np.array([(1 / (1 + np.exp( -(self.w.dot(x)) ))) for x in X])

    def predict(self, X: numpy.ndarray):
        """
        Predicts the labels for the given data using a threshold of 0.5 
        on the probabilities.

        Parameters
        ----------
        X : numpy.ndarray
            An array that contains the data points.

        Returns
        -------
        numpy.ndarray
            An array  that contains the predicted labels (-1 or 1) for 
            each data point.
        """
        pred = self.predict_prob(X)
        return np.where(pred >= 0.5, 1, -1)

    def get_w(self):
        """
        Returns the weight vector.

        Returns
        ----------
        w : numpy.ndarray
            The weight vector.
        """
        return self.w
    
    def set_w(self, w: list):
        """
        Sets the weight vector.

        Parameters
        ----------
        w : numpy.ndarray
            The weight vector.
        """
        self.w = w
