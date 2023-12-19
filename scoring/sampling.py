import torch
from scipy.stats import multivariate_normal
import random
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import numpy as np 


def sample_flow(ypred, flow, n_samples = 1000):
    ypred = ypred.permute(1, 0, 2).contiguous().view(5, -1).permute(1, 0)
    multi_distrib = torch.zeros(ypred.shape[0], n_samples, 2)
    with torch.no_grad():
        for i in tqdm(range(ypred.shape[0])):
            samples_gen = torch.tensor(flow.sample(n_samples, context = ypred[i, :].cpu().float()))
            multi_distrib[i, :, :] = samples_gen.reshape((1, n_samples, 2))
    return multi_distrib

def sample_gaussian(ypred, n_samples = 1000):
    eps = 1e-4
    ypred = ypred.permute(1, 0, 2).contiguous().view(5, -1).permute(1, 0)
    mean = ypred[:, :2]
    variables = ypred[:, 2:4]
    cov = ypred[:, 4]   

    cov = cov
    variables = variables + (variables < eps)*eps
    std =  torch.diag_embed(variables, dim1 = 1)
    std[:, 1, 0] = cov*torch.sqrt(variables[:, 0])*torch.sqrt(variables[:, 1])
    std[:, 0, 1] = std[:, 1, 0]
    std_matrix = std
    multi_distrib = torch.zeros((ypred.shape[0], n_samples, 2))
    with torch.no_grad():
        for i in tqdm(range(ypred.shape[0])):
            samples_gen = torch.tensor(multivariate_normal.rvs(mean[i, :].cpu().detach(), std_matrix[i, :, :].cpu().detach(), n_samples))
            multi_distrib[i, :, :] = samples_gen.reshape((1, n_samples, 2))
    return multi_distrib


def sample_gbm(y, ypred, quantiles, n_samples = 1000):
    y = y.permute(1, 0, 2).contiguous().view(2, -1).permute(1, 0)
    ypred = ypred.permute(1, 2, 0, 3).contiguous().view(2, 21, -1).permute(2, 0, 1)
    samples_u = torch.zeros((y.shape[0], n_samples))
    samples_v = torch.zeros((y.shape[0], n_samples))
    samples = torch.zeros((y.shape[0], n_samples, 2))
    with torch.no_grad():
        for t in tqdm(range(y.shape[0])):
            f_u = CubicSpline(quantiles, ypred[t, 0, :], bc_type='natural')
            f_v = CubicSpline(quantiles, ypred[t, 1, :], bc_type='natural')
            x_in_u = np.array([f_u(random.random()) for i in range(n_samples)]).reshape(n_samples)
            x_in_v = np.array([f_v(random.random()) for i in range(n_samples)]).reshape(n_samples)
            samples_u[t, :] = torch.tensor(x_in_u)
            samples_v[t, :] = torch.tensor(x_in_v)
            samples[t, :, 0] = samples_u[t, :]
            samples[t, :, 1] = samples_v[t, :]
    return samples