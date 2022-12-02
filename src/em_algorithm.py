from typing import List, Tuple

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score

from src.constants import NUM_STEPS_TO_PLOT, INITIAL_LOG_LIKELIHOOD_VAL, LOG_LIKELIHOOD_CONVERGENCE
from src.utils import plot_log_likelihood, plot_scatter_data

np.random.seed(2)


class GMM:
    """
    A Python implementation of Gaussian Mixture Model.
    GMM is a type of clustering algorithm, whereby each cluster is determined by a Gaussian distribution.
    The underlying assumption is that each data point could have been generated by the mixture of the distributions,
    with a corresponding probability to belong to each of the clusters.

    The Expectation Maximization (EM) Algorithm is used to find maximum likelihood estimates of parameters (for GMM the
    parameters are weights, means and covariance).
    The algorithm consists of two steps:
    - E-step: Estimate the probability of data points to belong to each distribution (denoted as z)
    - M-step: Update the value of the model parameters based on the estimation of z the E-step
    These two steps are repeated until convergence.
    """

    def __init__(self, n_clusters: int, max_iterations: int, X: np.ndarray):
        """
        Define a GMM with the given number of clusters and features.

        Parameters
        ----------
        n_clusters : int
            Number of clusters (represents the number of Gaussians as well).
        max_iterations : int
            Maximum number of iteration for the EM algorithm.
        X : ndarray
            Input data in shape of [n_samples, n_dims].
        """

        self.n_clusters = n_clusters
        self.max_iters = max_iterations
        self.X = X
        self.n_samples, self.n_dims = X.shape

        self.z = np.zeros((self.n_samples, self.n_clusters))
        self.mu = np.random.rand(n_clusters, self.n_dims)
        self.sigma = np.tile(np.eye(self.n_dims), (self.n_clusters, 1, 1))
        self.cluster_probabilities = np.ones(n_clusters) / n_clusters

    def calc_membership_in_cluster_probs(self) -> np.ndarray:
        """ Calculate the probability of each data point to belong to each cluster """
        return np.array(
            [self.calc_membership_in_single_cluster(i) for i in range(self.n_clusters)]).T  # [n_samples, n_clusters]

    def calc_membership_in_single_cluster(self, cluster_index: int) -> List[float]:
        """
        Calculate the probability of data points to belong to a single cluster (which is given by its index).

        Parameters
        ----------
        cluster_index : int
            The index of the cluster for which the probabilities are calculated.

        Returns
        -------
        List[float]
            A list containing normalized probabilities of the data points to belong to the given cluster.
        """
        return self.cluster_probabilities[cluster_index] * multivariate_normal.pdf(self.X,
                                                                                   mean=self.mu[cluster_index],
                                                                                   cov=self.sigma[cluster_index])

    def estimation_step(self):
        """
        Estimation step of EM algorithm.
        For each data point, estimate its probability to belong to each cluster.
        """
        # z is of shape [n_samples, n_clusters]
        # cluster probabilities are of shape [n_clusters]
        self.z = self.calc_membership_in_cluster_probs()
        self.z /= self.z.sum(axis=1, keepdims=True)

    def maximization_step(self):
        """
        Maximization step of EM algorithm.
        Update the model parameters (mu, sigma) by performing a single maximization step, as follows:
        mu_j = sum n_samples (z_ij * x_i) / sum n_samples (z_ij)
        sigma_j = sum n_samples z_ij * (x_i - mu_j)^2 / sum n_samples (z_ij)

            Where j is a cluster's index, and i is a sample's index

        The update steps are performed in a matrix-fashion rather the using for loops.
        """
        # z is of shape [n_samples, n_dims]
        # X is of shape [n_samples, n_dims]
        # mu is of shape [n_clusters, n_dims]
        # sigma is of shape [n_clusters, n_dims, n_dims]

        # update cluster_probabilities
        self.cluster_probabilities = self.z.mean(axis=0)

        # update mu: weighted average of cluster means
        self.mu = np.matmul(self.z.T, self.X) / self.z.sum(axis=0)[:, None]

        # update sigma: weighted squared error
        x_reshaped = np.tile(self.X[:, :, None], (1, 1, self.n_clusters)).transpose(
            [2, 1, 0])  # [n_clusters, n_dims, n_samples]
        mu_reshaped = np.tile(self.mu[:, :, None], (1, 1, self.n_samples))  # [n_clusters, n_dims, n_samples]
        z_reshaped = np.tile(self.z[:, :, None], (1, 1, self.n_dims)).transpose(
            [1, 2, 0])  # [n_clusters, n_dims, n_samples]
        centered_data = x_reshaped - mu_reshaped
        self.sigma = np.matmul((centered_data * z_reshaped), centered_data.transpose([0, 2, 1]))
        self.sigma /= self.z.sum(axis=0)[:, None, None]

    def calc_log_likelihood(self) -> float:
        """ Calculate the log likelihood with the estimated parameters """
        return np.mean([np.log(self.calc_membership_in_cluster_probs().sum(axis=0))])

    def run(self, plot_clusters: bool, plot_ll: bool) -> float:
        """
        Run EM algorithm iteratively, until convergence or getting to the defined max iterations, and calculate
        the silhouette score for the predicted labels.
        The function receives two boolean flags to decide whether to generate plots.

        Parameters
        ----------
        plot_clusters : bool
            If True, plot the assignment of the data points to clusters (every [NUM_STEPS_TO_PLOT] iterations).
        plot_ll : bool
            If True, plot the log-likelihood values.

        Returns
        ----------
        float
            Silhouette score
        """
        ll_values = []
        converged = False
        best_ll = INITIAL_LOG_LIKELIHOOD_VAL
        i = 0
        while not converged and i < self.max_iters:
            log_likelihood, y_pred = self.single_em_iteration()
            ll_values.append(log_likelihood)
            # check convergence of the algorithm
            converged = abs(log_likelihood - best_ll) <= LOG_LIKELIHOOD_CONVERGENCE

            # update the best log-likelihood value
            best_ll = log_likelihood if log_likelihood > best_ll else best_ll

            if plot_clusters:
                if i % NUM_STEPS_TO_PLOT == 0:
                    plot_scatter_data(self.X[:, 0], self.X[:, 1], y_pred, i)
            i += 1

        if plot_ll:
            plot_log_likelihood(ll_values)

        return silhouette_score(self.X, y_pred)

    def single_em_iteration(self) -> Tuple[float, np.ndarray]:
        """
        Execute a single iteration of EM algorithm.

        Returns
        -------
        Tuple[float, ndarray]
            Log-likelihood, y predictions
        """
        self.estimation_step()
        self.maximization_step()
        log_likelihood = self.calc_log_likelihood()
        y_pred = np.argmax(self.z, 1)  # each sample belongs to the cluster with the highest probability

        return log_likelihood, y_pred
