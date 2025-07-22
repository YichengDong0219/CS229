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
        mapped_input = self.create_sin(self.k, X) if self.sine else self.create_poly(self.k, X)
        self.theta = np.linalg.solve(mapped_input.T @ mapped_input, mapped_input.T @ y)
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
        return np.hstack([X[:,1][:, np.newaxis] ** i for i in range((k or 0)+1)])
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        return np.hstack([self.create_poly(k, X), np.sin(X)[:, 1][:, np.newaxis]])
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
        mapped_input = self.create_sin(self.k, X) if self.sine else self.create_poly(self.k, X)
        return mapped_input @ self.theta
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png', print_theta=False):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel(k=k, sine=sine)
        model.fit(train_x, train_y)
        plot_y = model.predict(plot_x)
        
        if print_theta:
            print(model.theta)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k, color=plt.cm.tab10(int(np.floor(int(np.random.random() * 11))))) # type: ignore

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # Experiment b)
    run_exp(train_path=train_path, ks=[3], filename="plot-degree3.png")
    
    # Experiment c)
    run_exp(train_path=train_path, ks=[3, 5, 10, 20], filename="plot-degrees[3, 5, 10, 20].png")
    
    # Experiment d)
    run_exp(train_path=train_path,sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename="plot-sine-degrees[0, 1, 2, 3, 5, 10, 20].png")

    # Experiment e)
    run_exp(train_path=small_path, ks=[1, 2, 5, 10, 20], filename="plot-small-degrees[1, 2, 5, 10, 20].png")

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
