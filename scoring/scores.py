import torch
import numpy as np
import random

def proba_scores(y, pred_samples, std_scaler, mean_scaler, bootstrap = False):

    n_samples = pred_samples.shape[1]
    y = y.permute(1, 0, 2).contiguous().view(2, -1).permute(1, 0)

    std_scaler = std_scaler.reshape(1, std_scaler.shape[0]).tile(y.shape[0], 1)
    mean_scaler = mean_scaler.reshape(1, mean_scaler.shape[0]).tile(y.shape[0], 1)
    y =  y*std_scaler + mean_scaler

    pred_samples = pred_samples.permute(1, 2, 0, 3).contiguous().view(n_samples, 2, -1).permute(2, 0, 1)
    std_scaler = std_scaler.reshape(std_scaler.shape[0], 1, std_scaler.shape[1]).tile(1, pred_samples.shape[1], 1)
    mean_scaler = mean_scaler.reshape(mean_scaler.shape[0], 1, mean_scaler.shape[1]).tile(1, pred_samples.shape[1], 1)

    pred_samples = pred_samples*std_scaler + mean_scaler

    mse = ((pred_samples.mean(axis = 1) - y)**2).mean(axis = 1)
    mae = (abs(pred_samples.median(axis = 1).values - y)).mean(axis = 1)

    crps = torch.zeros((y.shape[0], y.shape[1]))
    for i_var in range(y.shape[1]):
        ED = (abs(pred_samples[:, :, i_var] - y[:, i_var].unsqueeze(1).tile(1, n_samples))).mean(axis = 1)
        EI = (abs(pred_samples[:,:int(n_samples/2), i_var] - pred_samples[:,int(n_samples/2):, i_var])).mean(axis = 1)
        crps[:, i_var] = ED - 0.5*EI
    
    ED = torch.linalg.norm(pred_samples - y.unsqueeze(1).tile(1, n_samples, 1), dim = 2).mean(axis = 1)
    EI = torch.linalg.norm(pred_samples[:, :int(n_samples/2), :] - pred_samples[:, int(n_samples/2):, :], dim = 2).mean(axis = 1)
    es = ED - 0.5*EI

    diff = torch.sqrt(abs(y[:, 0].cpu().detach() - y[:, 1].cpu().detach()))
    VSp = (2*(diff - torch.sqrt(abs(pred_samples[:, :, 0].cpu().detach() - pred_samples[:, :, 1].cpu().detach())).mean(dim = 1))**2)

    error_per_timestep = {'es' : es, 'crps' : crps, 'mse': mse, 'vs' : VSp, 'mae' : mae}
    error_mean = {'es' : es.mean(), 'crps' : crps.mean(axis = 0), 'mse': mse.mean(), 'vs' : VSp.mean(), 'mae' : mae.mean()}
    return error_per_timestep, error_mean


def deterministic_scores(y, ypred, std_scaler, mean_scaler):
    std_scaler = std_scaler.reshape(1, std_scaler.shape[0], 1).tile(y.shape[0], 1, y.shape[-1])
    mean_scaler = mean_scaler.reshape(1, mean_scaler.shape[0], 1).tile(y.shape[0], 1, y.shape[-1])
    y = y*std_scaler + mean_scaler
    y = y.permute(1, 0, 2).contiguous().view(2, -1).permute(1, 0).detach()
    ypred = ypred.permute(1, 0, 2).contiguous().view(2, -1).permute(1, 0).detach()    
    mse = ((y - ypred)**2).mean(axis = 1)
    mae = abs((y - ypred)).mean(axis = 1)
    scores_deterministic = {'mse' : mse, 'mae' : mae}
    scores_mean = {'mse' : mse.mean(), 'mae' : mae.mean()}
    return scores_deterministic, scores_mean

def RMSE_global(mse, N_entries):
    return torch.sqrt(mse.reshape(N_entries, 60).mean(axis = 0)).mean()

def rank_histogram(samples, y):
    n_samples = samples.shape[1]
    n_vars = y.shape[1]
    samples = samples.permute(1, 2, 0, 3).contiguous().view(n_samples, n_vars, -1).permute(2, 0, 1)
    y = y.permute(1, 0, 2).contiguous().view(n_vars, -1).permute(1, 0)
    n_entries = y.shape[0]
    samples = torch.cat((samples, y.unsqueeze(1)), dim = 1)
    ranks = torch.zeros((n_entries, n_samples + 1))
    rank_final = torch.zeros((n_entries))
    for j in range(n_samples + 1):
        for d in range(n_vars):
            if d == 0:
                idx = (samples[:, :, d] < samples[:, j, d].unsqueeze(1))
            else:
                idx = idx*(samples[:, :, d] < samples[:, j, d].unsqueeze(1))
        ranks[:, j] = idx.sum(axis = 1)
    s_inf = (ranks < ranks[:, -1].unsqueeze(1)).sum(axis = 1)
    s_eq = (ranks  == ranks[:, -1].unsqueeze(1)).sum(axis = 1)
    for t in range(y.shape[0]):
        rank_final[t] = random.randint(s_inf[t] + 1, s_inf[t] + s_eq[t])/(n_samples+1)    

    quantiles = [0.1*i for i in range(1, 11)]
    n_quantiles = len(quantiles)
    n_tot_samples = rank_final.shape[0]
    n_samples_quant = 0
    n_samples_quant_cumsum = 0
    rel_idx = 0
    for i, quantile in enumerate(quantiles):
        n_samples_quant = (rank_final < quantile).sum() - n_samples_quant_cumsum
        n_samples_quant_cumsum = n_samples_quant_cumsum + n_samples_quant
        f_quant = n_samples_quant / n_tot_samples
        rel_idx = rel_idx + abs(f_quant - 1/n_quantiles)/n_quantiles*100
    return rank_final, rel_idx