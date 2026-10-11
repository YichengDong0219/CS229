import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import util

def generate_plot(betas, X, Y, X_val, Y_val, save_path):
    """Generate a scatter plot of validation error vs. norm

    Args:
        betas: list of numpy arrays, indicating different solutions
        X, Y: training dataset 
        X_val, Y_val: validataion dataset
        save_path: path to save the plot
    """
    # check if the validationerror is zero
    for b in betas:
        assert(np.allclose(X.dot(b), Y))

    # compute the norm and the validation error of all the solutions in list beta
    val_err = []
    norms = []
    for i in range(len(betas)):
        val_err.append(np.mean((X_val.dot(betas[i]) - Y_val) ** 2))
        norms.append(np.linalg.norm(betas[i]))

    print(norms)
    # plot the validation error against norm of the solution
    util.plot_points(norms, val_err, save_path)

def linear_model_main():
    # *** START CODE HERE ***
    raise NotImplementedError("TODO: implement linear_model_main")
    # *** END CODE HERE ***
    
    # you can use the function generate_plot()
    # *** END CODE HERE ***

if __name__ == '__main__':
    linear_model_main()
