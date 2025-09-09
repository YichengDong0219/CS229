import numpy as np
import util
import sys
from random import random
import matplotlib.pyplot as plt


sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(validation_path, add_intercept=True)
    positive_x_train = x_train[y_train == 1]
    negative_x_train = x_train[y_train == 0]
    positive_x_val = x_val[y_val == 1]
    negative_x_val = x_val[y_val == 0]
    
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    vanilla_logr = LogisticRegression()
    vanilla_logr.fit(x_train, y_train)
    preds = vanilla_logr.predict(x_val)
    
    a_0 = (np.sum(preds[y_val == 0] < 0.5)) / len(y_val[y_val == 0])
    a_1 = (np.sum(preds[y_val == 1] >= 0.5)) / len(y_val[y_val == 1])
    a = (1 / (1 + 1 / kappa)) * a_1 + (1 / (1 + kappa)) * a_0
    bal_a = 0.5 *(a_1 + a_0)
    
    print(f"A1: {a_1}, A0: {a_0}, A: {a}, bal. A: {bal_a}")
    
    np.savetxt(output_path_naive, preds)
    
    # Figure
    plt.figure()
    plt.scatter(negative_x_val[:, 1], negative_x_val[:, 2], marker='o', label='Class 0') # type: ignore
    plt.scatter(positive_x_val[:, 1], positive_x_val[:, 2], marker='s', label='Class 1') # type: ignore
    
    x_boundary = np.array([np.min(x_val[:, 1]), np.max(x_val[:, 1])])

    plt.plot(x_boundary, vanilla_logr.boundary(x_boundary), color='red')

    plt.savefig(output_path_naive.split('.')[0] + '.png')
    
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    upsample_x = np.concatenate([negative_x_train, np.tile(positive_x_train, (int(np.ceil(1/kappa)), 1))])
    upsample_y = np.concatenate([np.zeros((negative_x_train.shape[0])), np.ones((positive_x_train.shape[0] * int(np.ceil(1/kappa))))])
    
    upsample_logr = LogisticRegression()
    upsample_logr.fit(upsample_x, upsample_y)
    preds = upsample_logr.predict(x_val)
    
    a_0 = (np.sum(preds[y_val == 0] < 0.5)) / len(y_val[y_val == 0])
    a_1 = (np.sum(preds[y_val == 1] >= 0.5)) / len(y_val[y_val == 1])
    a = (1 / (1 + 1 / kappa)) * a_1 + (1 / (1 + kappa)) * a_0
    bal_a = 0.5 *(a_1 + a_0)
    
    print(f"A1: {a_1}, A0: {a_0}, A: {a}, bal. A: {bal_a}")
    
    np.savetxt(output_path_upsampling, preds)
    
    # Figure
    plt.figure()
    plt.scatter(negative_x_val[:, 1], negative_x_val[:, 2], marker='o', label='Class 0') # type: ignore
    plt.scatter(positive_x_val[:, 1], positive_x_val[:, 2], marker='s', label='Class 1') # type: ignore
    
    x_boundary = np.array([np.min(x_val[:, 1]), np.max(x_val[:, 1])])

    plt.plot(x_boundary, upsample_logr.boundary(x_boundary), color='red')

    plt.savefig(output_path_upsampling.split('.')[0] + '.png')
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
