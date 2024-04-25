"""Implementations of Conditionnal Multivariate Normal distribution as base distribution for nflows.
Addition to the existing package. This proposed function is only implemented for 2D distributions """

import numpy as np
import torch
from torch import nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from scipy.stats import multivariate_normal


class MultivariateNormal(Distribution):
    """A multivariate Normal"""

    def __init__(self, shape, context_encoder = None, cov_description = 'pearson',
                 **kwargs):
        super().__init__()
        self._shape = torch.Size(shape)
        self.cov_description = cov_description
        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder
        
        self.register_buffer("_log_z",
                             torch.tensor(0.5 * shape[0] * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _compute_params(self, context):
        return self._context_encoder(context)
    
    
    def build_cov_matrix(self, pred_vector):
        n_pred = pred_vector.shape[1]
        dim = int((-3 + torch.sqrt(torch.tensor(9 + 8*n_pred)))/2)
        n_mu = dim
        n_sig = dim
        n_rho = int((dim*(dim-1))/2)
        mean = pred_vector[:, :n_mu]
        diag_elts = pred_vector[:, n_mu:n_mu+n_sig]
        non_diag_elts = pred_vector[:, n_mu+n_sig:n_mu+n_sig+n_rho]

        diag_elts = diag_elts + (diag_elts < 0.001)*0.001
        if len(pred_vector.shape) == 3:
            n_timesteps = pred_vector.shape[-1]
            n_entries = pred_vector.shape[0]
            diag_elts = diag_elts.permute(1, 0, 2).contiguous().view(n_sig, -1).permute(1, 0)
            non_diag_elts = non_diag_elts.permute(1, 0, 2).contiguous().view(n_rho, -1).permute(1, 0)

        L = torch.diag_embed(diag_elts, dim1=1)
        i_rho = 0
        for i in range(dim):
            for j in range(i+1, dim):
                L[:, i, j] = non_diag_elts[:, i_rho]
                i_rho+=1
        cov_matrix = torch.matmul(L, torch.transpose(L, 1, 2))
        if len(pred_vector.shape) == 3:
            cov_matrix = cov_matrix.reshape(n_entries, n_timesteps, dim, dim).permute(0, 2, 3, 1)

        return mean, cov_matrix

    def _log_prob(self, inputs, context):
        
        params = self._compute_params(context)
    
        mu, cov_matrix = self.build_cov_matrix(params)

        det_cov = torch.linalg.det(cov_matrix)
        log_cov = 0.5*torch.log(det_cov)
        inv_Sigma = torch.linalg.inv(cov_matrix)
        inputs = inputs.unsqueeze(2)
        mu = mu.unsqueeze(2)
        exp_term = 0.5 * torch.matmul(torch.matmul(torch.transpose(inputs - mu, 1, 2), inv_Sigma), (inputs - mu))
        log_z = torch.tile(self._log_z, log_cov.size())

        proba_vector = log_z + log_cov + exp_term.flatten()
        proba_vector = proba_vector.flatten()
        return -proba_vector
    
    def _sample(self, num_samples, context, scenarios_sample = False, temp_cov = None):
        params = self._compute_params(context)
        mu, cov_matrix = self.build_cov_matrix(params)
        samples = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov_matrix).rsample(torch.Size([num_samples])).permute(1, 0, 2)
        return samples.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)

