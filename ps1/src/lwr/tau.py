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
    # *** START CODE HERE ***
    raise NotImplementedError("TODO: implement main")
    # *** END CODE HERE ***

    # *** END CODE HERE ***

def save_plot(fig, tau):
    # *** START CODE HERE ***
    raise NotImplementedError("TODO: implement save_plot")
    # *** END CODE HERE ***

def calculate_mse(y, pred):
    # *** START CODE HERE ***
    raise NotImplementedError("TODO: implement calculate_mse")
    # *** END CODE HERE ***
    
if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
