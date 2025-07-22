import os

import matplotlib.pyplot as plt
import numpy as np
import util
from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
        
    # *** START CODE HERE ***
    # Load Validation Set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    all_predictions = []
    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau)
        lwr.fit(x_train, y_train)
        
        predictions = lwr.predict(x_val)
        
        mse_val = calculate_mse(y_val, predictions)
        print(f"τ: {tau}; Mean Square Errror (Validation): {mse_val}")
        all_predictions.append(mse_val)
        
        # Plot!
        fig = plt.figure()
        plt.scatter(x_train[:, 1], y_train, c='b', marker='x', label='Train') # type: ignore
        plt.scatter(x_val[:, 1], predictions, c='r', marker='o', label='Validation') # type: ignore
        save_plot(fig, tau=tau)
        plt.close(fig)
    
    # Best tau value:
    best_tau_idx = np.argmin(all_predictions)
    best_tau = tau_values[best_tau_idx]
    lwr = LocallyWeightedLinearRegression(best_tau)
    
    # MSE Performance on test split
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    lwr.fit(x_train, y_train)
    best_tau_pred = lwr.predict(x_test)
    best_tau_mse = calculate_mse(y_test, best_tau_pred)
    print(f"[BEST] τ: {best_tau}; Mean Square Errror (Test): {best_tau_mse}")
    
    # Save predictions to pred_path
    np.savetxt(pred_path, best_tau_pred.flatten())

    # *** END CODE HERE ***

def save_plot(fig, tau):
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    fig_path = os.path.join(plots_dir, f"plot-tau-{tau}.png")
    fig.savefig(fig_path)

def calculate_mse(y, pred):
    return np.mean((pred - y)**2)
    
if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
