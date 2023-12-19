"""Implementations of Conditionnal Multivariate Normal distribution as base distribution for nflows.
Addition to the existing package. This proposed function is only implemented for 2D distributions """

import numpy as np
import torch
from torch import nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from scipy.stats import multivariate_normal


class MultivariateNormal(Distribution):
    """A 2D multivariate Normal"""

    def __init__(self, shape, context_encoder = None, cov_description = 'pearson'):
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
        return context
    
    def _log_prob(self, inputs, context):
        
        params = self._compute_params(context)

        mu = params[:, :2]
        variances = params[:, 2:4]
        pearsonCoeff = params[:, -1]

        std = torch.diag_embed(variances)
        std[:, 1, 0] = pearsonCoeff * \
            torch.sqrt(variances[:, 0]) * \
            torch.sqrt(variances[:, 1])
        std[:, 0, 1] = std[:, 1, 0]
        std_matrix = std

        det_cov = torch.linalg.det(std_matrix)
        log_cov = 0.5*torch.log(det_cov)
        inv_Sigma = torch.linalg.inv(std_matrix)

        inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
        mu = torch.reshape(mu, (inputs.shape[0], inputs.shape[1], 1))
        exp_term = 0.5 * torch.matmul(torch.matmul(torch.transpose(inputs - mu, 1, 2), inv_Sigma), (inputs - mu))
        log_z = torch.tile(self._log_z, log_cov.size())

        proba_vector = log_z + log_cov + exp_term.flatten()
        proba_vector = proba_vector.flatten()
        return -proba_vector

    def _sample(self, num_samples, context):
        mean = context[:2]
        variances = context[2:4]
        pearsonCoeff = context[-1]
        
        std = torch.diag_embed(variances)
        std[1, 0] = pearsonCoeff * \
            torch.sqrt(variances[0]) * \
            torch.sqrt(variances[1])
        std[0, 1] = std[1, 0]
        std_matrix = std  
        
        samples = torch.tensor(multivariate_normal.rvs(mean, std_matrix, num_samples)).float()
        return samples

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)
