import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train, y_train)
    
    # Get MSE value on the evaluation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    predictions = lwr.predict(x_val)
    
    mse_val = np.mean((predictions - y_val)**2)
    print(f"Mean Square Errror (Validation): {mse_val}")
    
    # Plot validation predictions on top of training set
    plt.figure()
    plt.scatter(x_train[:, 1], y_train, c='b', marker='x', label='Train')  # type: ignore 
    plt.scatter(x_val[:, 1], predictions, c='r', marker='o', label='Validation')    # type: ignore data
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('LWR: Training and Validation Data')
    plt.show()
    
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None
        self.last_theta = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        predictions = []
        
        for input_x in x:
            # Calculate distances between input_x and each row in self.x
            distances = np.linalg.norm(self.x - input_x, axis=1)
            
            # Create diagonal weight matrix using the calculated distances
            weights = np.diag(np.exp(-1 * (distances ** 2 / (2 * self.tau ** 2))))
            
            # Calculate theta using the weighted normal equation
            self.last_theta = np.linalg.inv(self.x.T @ weights @ self.x) @ self.x.T @ weights @ self.y # type: ignore
            
            # Make prediction for this input point
            predictions.append(self.last_theta.T @ input_x)
            
        return np.array(predictions)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
