import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    
    n, dim = x.shape
    
    shuffled_indices = np.random.permutation(x_all.shape[0])
    x_shuffled = x_all[shuffled_indices]
    
    groups = np.array_split(x_shuffled, K)
    mu = [np.mean(group, axis=0) for group in groups]
    sigma = [np.cov(group.T) for group in groups]

    phi = np.full(K, 1/K)
    w = np.full((n, K), 1/K)


    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """

    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        
        
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        prev_ll = ll
        n, _ = x.shape
        
        p_x_given_z = calculate_p_x_given_z(x, mu, sigma, K)

        # Calculate the numerator of the responsibility
        numerator_res = p_x_given_z * phi
        denominator_res = np.sum(numerator_res, axis=1, keepdims=True)

        w = numerator_res / denominator_res

        # (2) M-step: Update the model parameters phi, mu, and sigma
        # Sum of responsibilities for each cluster
        sum_w_j = np.sum(w, axis=0)
        mu_numerator = np.dot(w.T, x)
        mu = [mu_numerator[j] / sum_w_j[j] for j in range(K)]
        
        phi = sum_w_j / n

        sigma = []
        for j in range(K):
            diff = x - mu[j]
            sigma_num = np.dot(diff.T, np.diag(w[:, j])).dot(diff)
            sigma.append(sigma_num / sum_w_j[j])


        # (3) Compute the log-likelihood of the data to check for convergence.
        ll = np.sum(np.log(denominator_res))

        
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        it += 1
        # *** END CODE HERE ***

    # Print the number of iterations and final log-likelihood
    print(f'Converged in {it} iterations')
    print(f'Final log-likelihood: {ll:.4f}')
    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        
        
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        prev_ll = ll
        n, dim = x.shape
        
        p_x_given_z = calculate_p_x_given_z(x, mu, sigma, K)

        # Calculate the numerator of the responsibility
        numerator_res = p_x_given_z * phi
        denominator_res = np.sum(numerator_res, axis=1, keepdims=True)

        w = numerator_res / denominator_res
        
        # (2) M-step: Update the model parameters phi, mu, and sigma
        n_tilde, _ = x_tilde.shape

        sum_w_j = np.sum(w, axis=0)
        z_tilde_counts = np.zeros(K)
        for i in range(n_tilde):
            z_tilde_counts[int(z_tilde[i])] += 1

        # Phi
        phi = (sum_w_j + alpha * z_tilde_counts) / (n + alpha * n_tilde)

        # Mu
        mu_num_unlabeled = np.dot(w.T, x)
        mu_num_labeled = np.zeros((K, dim))
        for i in range(n_tilde):
            mu_num_labeled[int(z_tilde[i])] += x_tilde[i]
        
        mu_denominator = sum_w_j + alpha * z_tilde_counts
        mu = (mu_num_unlabeled + alpha * mu_num_labeled) / mu_denominator[:, np.newaxis]
        mu = [mu[j,:] for j in range(K)]

        # Sigma
        sigma = []
        for j in range(K):
            # Unlabeled part
            diff_unlabeled = x - mu[j]
            sigma_num_unlabeled = np.dot(diff_unlabeled.T, np.diag(w[:, j])).dot(diff_unlabeled)

            # Labeled part
            diff_labeled = x_tilde - mu[j]
            z_tilde_j_indices = (z_tilde == j).squeeze()
            sigma_num_labeled = np.dot(diff_labeled[z_tilde_j_indices].T, diff_labeled[z_tilde_j_indices])
            
            sigma_j = (sigma_num_unlabeled + alpha * sigma_num_labeled) / mu_denominator[j]
            sigma.append(sigma_j)
        
        # (3) Compute the log-likelihood of the data to check for convergence.
        ll_unlabeled = np.sum(np.log(denominator_res))
        
        # LL for Labeled Data
        p_xtilde_given_ztilde = np.zeros((n_tilde, K))
        for j in range(K):
            diff = x_tilde - mu[j]
            p_xtilde_given_ztilde[:, j] = (1. / (np.power(2 * np.pi, dim / 2.) * np.sqrt(np.linalg.det(sigma[j])))) * \
                                 np.exp(-0.5 * np.sum(np.dot(diff, np.linalg.inv(sigma[j])) * diff, axis=1))
        
        # Calculating Joint Loss
        p_joint_labeled = p_xtilde_given_ztilde * phi
        ll_labeled = 0
        for i in range(n_tilde):
            ll_labeled += np.log(p_joint_labeled[i, int(z_tilde[i])])
        
        ll = ll_unlabeled + alpha * ll_labeled
        
        it += 1
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    print(f'Converged in {it} iterations')
    print(f'Final log-likelihood: {ll:.4f}')
    return w


# *** START CODE HERE ***
# Helper functions
def calculate_p_x_given_z(x, mu, sigma, K):
    n, dim = x.shape
    p_x_given_z = np.zeros((n, K))
    
    for j in range(K):
        det_sigma = np.linalg.det(sigma[j])
        inv_sigma = np.linalg.inv(sigma[j])
        norm_const = 1.0 / (np.power(2 * np.pi, float(dim) / 2) * np.sqrt(det_sigma))
        diff = x - mu[j]
        exponent = -0.5 * np.sum(np.dot(diff, inv_sigma) * diff, axis=1)
        p_x_given_z[:, j] = norm_const * np.exp(exponent)
    
    return p_x_given_z
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=True, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
