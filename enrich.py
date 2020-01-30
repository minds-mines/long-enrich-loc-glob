"""
Longitudinal Enrichment balancing between global and local patterns with p-order
"""

import numpy as np
import scipy
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import norm, sqrtm
from scipy.sparse.linalg import eigs
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# NOTE: Commented code is for debugging purposes if needed.


class LongEnrichment():

    def __init__(self, r, p, delta_gamma=0.01, delta_theta=0.01):
        """Create a transformer for TER_PCA_LE

        Arguments:
            r {int} -- the output dimensions of the learned embedding
            p {float} -- sensitivity to outliers, 2 being highest sensitivity and closer to 0 being lower sensitivity

        Keyword Arguments:
            delta_gamma {float} -- [description] (default: {0.001})
            delta_theta {float} -- [description] (default: {0.001})
        """
        self.r = r
        self.p = p
        self.delta_gamma = delta_gamma
        self.delta_theta = delta_theta
        self.objectives = []
        self.W = None

    def fit(self, instances):
        """Learn the W for a group of datapoints
        
        Args:
            instances (iterable): The samples from which to learn a projection
        """
        # Normalize the data
        # all_vals = np.vstack([bag, instances])
        # scaled_vals = MinMaxScaler().fit_transform(all_vals.T)

        # bag = scaled_vals.T[0,:]
        # instances = scaled_vals.T[1:, :]

        self.n_i, self.d = instances.shape

        # Initialize W
        M = np.random.uniform(0, 100, size=(self.d, self.r))
        W, _, _ = np.linalg.svd(M, full_matrices=False)
        assert W.shape == (self.d, self.r)
        assert np.allclose(W.T @ W, np.identity(self.r))

        # Determine the graph similarity matrix
        original_S = squareform(pdist(instances))
        S = original_S
        assert S.shape == (self.n_i, self.n_i)
        
        # original_lambda = self._calculate_objective(W, instances, S)

        # Loop until convergence
        n = 20 if self.p != 2 else 2
        for i in range(n):
            prevW = W
            # Calculate lambda
            lambda_, t, b = self._calculate_objective(W, instances, S)
            
            gamma = self._calculate_gamma(W, instances, S)
            
            # Apply updates to all variables dependent on gamma
            S = gamma @ original_S

            # Calculate the Laplacian matrix
            D = np.diag(S.sum(axis=1))
            L = D - S #+ S.sum()/(self.n_i ** 2) * np.ones((self.n_i, self.n_i))
            assert L.shape == (self.n_i, self.n_i)

            theta = lambda_ * self._calculate_theta(W, instances)

            # Update W
            value = instances.T @ L @ instances - instances.T @ theta @ instances

            # if np.allclose(value.real, value.real.T, atol=1e-8):
            #     print(f"Value is symmetric.")

            try:
                ev, W = eigs(value.real, k=self.r, which="SR", tol=0.01, maxiter=2000)
            except Exception:
                print(traceback.format_exc())
                break
            W = W.real
            # constraint = W.T @ W

            # mask = np.ones(constraint.shape, dtype=bool)
            # np.fill_diagonal(mask, 0)
            # constraint_max = constraint[mask].max()

            # if np.allclose(constraint.diagonal()[:-1], 1, atol=0.1) and constraint_max < 0.01:
            #     print(f"W is orthogonal at iteration {i}")
            # else:
            #     print(f"W is not orthogonal at iteration {i}")

            # convergenceDif = norm(W - prevW)
            # if convergenceDif < 0.001:
            #     print(f"W Converged after {i} iterations")
            #     if self.p == 2:
            #         break
            #     elif i > 50:
            #         break

            

            # if i >= 1:
            #     objConvergenceDif = abs(prev_lambda - lambda_)
            #     if objConvergenceDif < 0.01:
            #         print(f"Objective converged after {i} iterations")
            #         if i >= 5:
            #             break
            prev_lambda = lambda_

        self.W = W

    def transform(self, bag):
        """Calculate enriched representation of a bag
        
        Args:
            bag (iterable): The data representation of the bag
        
        Raises:
            Error: The model must be fit before running this method
        
        Returns:
            array: The enriched representation
        """
        
        if self.W is None:
            raise Error("Must fit data before transforming.")

        return self.W.T @ bag

    def transform_one(self, bag, instances):
        """Transform a bag and instances to the TER-PCA-LE

        Arguments:
            bag {np.ndarray} -- The 1*d representation of the data
            instances {np.ndarray} -- The n_i * d representation of the instances

        """
        self.fit(instances)
        return self.transform(bag)

    def _calculate_objective(self, W, X, S):
        """Calculate the lambda value

        Arguments:
            W {np.ndarray} -- The weight matrix
            X {np.ndarray} -- The instances
            S {np.ndarray} -- The similarity matrix of the instances

        Returns:
            float -- The objective value
        """
        top = np.sum([np.sum([S[i, j] * (np.linalg.norm(W.T@X[i, :] - W.T@X[j, :])
                              ** self.p) for j in range(self.n_i)]) for i in range(self.n_i)])
        bottom_1 = np.sum([np.linalg.norm(X[i, :]) **
                           self.p for i in range(self.n_i)])
        bottom_2 = np.sum([np.linalg.norm(X[i, :] - W @ W.T @ X[i, :]) ** self.p for i in range(self.n_i)])

        # assert (bottom_1 - bottom_2) >= -0.1

        objective = top/(bottom_1 - bottom_2)

        self.objectives.append(objective)

        return objective, top, bottom_1-bottom_2

    def _calculate_gamma(self, W, X, S):
        """Calculate the gamma matrix

        Arguments:
            W {np.ndarray} -- The weight matrix
            X {np.ndarray} -- The instances
            S {np.ndarray} -- The similarity matrix of the instances

        Returns:
            np.ndarray -- Gamma
        """
        gamma = np.diag([(self.p / 2)*(np.sum([S[i, j] * np.linalg.norm(W.T@X[i, :] - W.T@X[j, :]) **2
                                               for i in range(self.n_i)]) + self.delta_gamma)**((self.p-2)/2) for j in range(self.n_i)])

        assert gamma.shape == (self.n_i, self.n_i)
        return gamma

    def _calculate_theta(self, W, X):
        """Calculate theta tilde

        Arguments:
            W {np.ndarray} -- The weight matrix
            X {np.ndarray} -- The instances
        """
        theta = np.diag([(self.p / 2)*(np.linalg.norm(X[j, :] - W @ W.T @ X[j, :]) +
                                       self.delta_theta)**((self.p-2)/2) for j in range(self.n_i)])

        assert theta.shape == (self.n_i, self.n_i)
        return theta
