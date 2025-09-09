import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # Fit a Poisson Regression model
    
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_val)
    np.savetxt(save_path, predictions)
    
    # Plot the true count and predicted
    plt.figure()
    plt.scatter(y_val, predictions, color='blue', label='Predicted vs True')
    plt.xlabel('True Count')
    plt.ylabel('Predicted Expected Count')
    plt.legend()
    # Set both axes to start at 0 and have the same limits for proportionality
    min_val = 0
    max_val = max(np.max(y_val), np.max(predictions))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path + ".png")
    # *** END CODE HERE ***



class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        for i in range(self.max_iter):
            if self.theta is None:
                self.theta = np.zeros(x.shape[1])
            
            gradient = x.T @ (y - np.exp(x @ self.theta))
            
            new_theta = self.theta + self.step_size * gradient
            
            if np.sqrt(np.sum((new_theta - self.theta) ** 2)) < self.eps :
                break
            
            self.theta = new_theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
