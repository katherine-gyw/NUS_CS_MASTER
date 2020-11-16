""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    prob_ls = [np.zeros([len(x), n_states]) for x in x_list]
    for i in range(len(x_list)):
        for n in range(n_states):
            prob_ls[i][:, n] = scipy.stats.norm(loc=phi["mu"][n], scale=phi["sigma"][n]).pdf(x_list[i])
    for i, x in enumerate(x_list):
        # Forward
        alpha = np.tile(np.zeros(n_states), (len(x), 1))
        r = np.zeros(len(x))
        for k in range(n_states):
            alpha[0, k] = (
                    prob_ls[i][0, k]
                    # scipy.stats.norm(loc=phi["mu"][k], scale=phi["sigma"][k]).pdf(x[0])
                    * pi[k]
            )
        r[0] = alpha[0].sum()
        alpha[0] /= r[0]

        for t in range(1, len(x)):
            for k in range(n_states):
                alpha[t, k] = \
                    prob_ls[i][t, k] * np.dot(A[:, k], alpha[t - 1])
                    # scipy.stats.norm(loc=phi["mu"][k], scale=phi["sigma"][k]).pdf(x[t]) \
            r[t] = alpha[t].sum()
            alpha[t] /= r[t]

        # Backward
        beta = np.tile(np.zeros(n_states), (len(x), 1))
        beta[-1] = 1.0

        for t in range(len(x) - 2, -1, -1):
            for k in range(n_states):
                for j in range(n_states):
                    beta[t, k] += (
                            A[k, j]
                            * prob_ls[i][t+1, j] * beta[t + 1, j]
                            # * scipy.stats.norm(loc=phi["mu"][j], scale=phi["sigma"][j]).pdf(x[t + 1])
                    )
            beta[t] /= r[t + 1]

        # Singleton Marginal
        gamma_list[i] = alpha * beta

        # Pairwise Marginal
        for t in range(1, len(x)):
            for k in range(n_states):
                xi_list[i][t - 1, :, k] = (
                        A[:, k]
                        * prob_ls[i][t, k]
                        # * scipy.stats.norm(loc=phi["mu"][k], scale=phi["sigma"][k]).pdf(x[t])
                        * alpha[t - 1]
                        * beta[t, k]
                )
            xi_list[i][t - 1] /= r[t]
    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    # pi
    for n in range(n_states):
        a, b = 0, 0
        for i in range(len(x_list)):
            a += gamma_list[i][0, n]
            b += sum(gamma_list[i][0,:])
        pi[n] = a/b

    # A
    for j in range(n_states):
        for k in range(n_states):
            c, d = 0, 0
            for i in range(len(x_list)):
                c += sum(xi_list[i][:, j, k])
                d += sum(xi_list[i][:, j, :].flatten())
            A[j, k] = c/d

    # phi
    for n in range(n_states):
        # mu
        a, b = 0, 0
        for i in range(len(x_list)):
            a += sum(gamma_list[i][:,n] * x_list[i])
            b += sum(gamma_list[i][:,n])
        phi['mu'][n] = a/b

        # sigma
        c, d = 0, 0
        for i in range(len(x_list)):
            c += sum(gamma_list[i][:,n] * (x_list[i]-phi['mu'][n]) * np.transpose(x_list[i]-phi['mu'][n]))
            d += sum(gamma_list[i][:,n])
        phi['sigma'][n] = np.sqrt(c/d)
    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    prev_mu = None
    while True:
        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        pi, A, phi = m_step(x_list, gamma_list, xi_list)
        if prev_mu is None:
            prev_mu = phi['mu']
        elif np.mean(abs(prev_mu - phi['mu']))<0.0001:
            break
        else:
            print('Current average change for mu: {}, target: 0.0001.'.format(np.mean(abs(prev_mu - phi['mu']))))
            prev_mu = phi['mu']
    return pi, A, phi
