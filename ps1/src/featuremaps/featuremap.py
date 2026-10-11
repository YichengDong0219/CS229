import matplotlib.pyplot as plt
import numpy as np
import util
import random

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None, k=None, sine=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.k = k
        self.sine = sine

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement fit")
        # *** END CODE HERE ***
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement create_poly")
        # *** END CODE HERE ***
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement create_sin")
        # *** END CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement predict")
        # *** END CODE HERE ***
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png', print_theta=False):
    # *** START CODE HERE ***
    raise NotImplementedError("TODO: implement run_exp")
    # *** END CODE HERE ***


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    raise NotImplementedError("TODO: implement main")
    # *** END CODE HERE ***

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
