import torch 
import numpy as np
import random

def build_cov_matrix(pred, dim = 2):
        """Generates mean and covariance matrices from a predicted vector
           Covariance matrices are computed using the pearson description. 
           For output dimensions > 2, the cholesky decomposition should be prefered.

        Args:
            pred (torch tensor): The output of the forecast model
            dim (int, optional): Dimension of the target. Defaults to 2.

        Returns:
            mu (torch tensor): The mean vector of the multivariate Gaussian distributions
            std_matrix (torch tensor): The covariance matrices of the multivariate Gaussian distributions
    
        """
        eps = 1e-3
        n_mu = dim
        n_sig = dim
        n_rho = int((dim*(dim-1))/2)

        mu = pred[:, :n_mu, :]
        variances = pred[:, n_mu:n_mu+n_sig, :]
        pearsonCoeff = pred[:, n_mu+n_sig:n_mu+n_sig+n_rho, :].squeeze()

        variances = variances + (variances < eps)*eps
        std = torch.diag_embed(variances.permute(0, 2, 1), dim1=2).permute(0, 2, 3, 1)

        variances = variances + (variances < eps)*eps
        std = torch.diag_embed(variances.permute(
            0, 2, 1), dim1=2).permute(0, 2, 3, 1)
        std[:, 1, 0, :] = pearsonCoeff * \
            torch.sqrt(variances[:, 0, :]) * \
            torch.sqrt(variances[:, 1, :])
        std[:, 0, 1, :] = std[:, 1, 0, :]
        std_matrix = std

        return mu, std_matrix


def rank_histogram_2D(samples, y):
    n_t, n_samples, n_var, n_tl  = samples.shape
    samples = samples.permute(1, 2, 0, 3).contiguous().view(n_samples, n_var, -1).permute(2, 0, 1)
    y = y.permute(1, 0, 2).contiguous().view(n_var, -1).permute(1, 0)
    rank_final = np.zeros((samples.shape[0]))
    samples = samples[:, :n_samples, :]
    for t in range(y.shape[0]):
        samples_t = np.append(samples[t, :, :], np.reshape(y[t, :], (1, 2)), axis = 0)
        rank = np.zeros((n_samples + 1, 1))
        for j in range(n_samples + 1): 
            rank[j] = (((samples_t[:, 0] < samples_t[j, 0])*(samples_t[:, 1] < samples_t[j, 1]))*1).sum()
        s_inf = (rank[:] < rank[-1]).sum()
        s_eq = (rank[:]  == rank[-1]).sum()
        rank_final[t] = random.randint(s_inf + 1, s_inf + s_eq)/n_samples
    return rank_final