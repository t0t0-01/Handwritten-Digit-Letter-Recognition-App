import numpy as np

class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """ 
        N = []
        # Euclidian distance
        euc = np.sqrt(np.sum(np.power(self.x_train-x, 2), axis=1))
        # Manhattan Distance
        # man = np.sum(np.abs(np.subtract(x,i)))
        ind = np.argsort(euc)  
        classes = self.y_train[ind[:self.k]] 
        return classes
