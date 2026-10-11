import matplotlib.pyplot as plt
import numpy as np
import util


def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
    """
    # *** START CODE HERE ***
    raise NotImplementedError("TODO: implement main")
    # *** END CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True, reg_rate=0.0):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        self.reg_rate = reg_rate
        
    
    def boundary(self, x):
        """
        Outputs the y-values of the decision boundary (p(y|x) = 0.5) for the input x-values
        """    
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement boundary")
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement fit")
        # *** END CODE HERE ***
        # *** END CODE HERE ***
    
    def loss(self, x,y):
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement loss")
        # *** END CODE HERE ***
    
    def loss_gradient(self, x,y):
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement loss_gradient")
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        raise NotImplementedError("TODO: implement predict")
        # *** END CODE HERE ***
        # *** END CODE HERE ***

if __name__ == '__main__':
    print('==== Training model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a_no-reg.png')

    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b_no-reg.png')
