import torch
from tqdm import tqdm
import numpy as np
import random
from tqdm import tqdm


import numpy as np
def proba_scores(y, pred_samples, std_scaler, mean_scaler):

    n_samples = pred_samples.shape[1]
    y = y.permute(1, 0, 2).contiguous().view(2, -1).permute(1, 0)

    pred_samples = pred_samples.permute(1, 2, 0, 3).contiguous().view(n_samples, 2, -1).permute(2, 0, 1)

    std_scaler = std_scaler.unsqueeze(0).tile(y.shape[0], 1)
    mean_scaler = mean_scaler.unsqueeze(0).tile(y.shape[0], 1)
    y =  y*std_scaler + mean_scaler

    std_scaler = std_scaler.unsqueeze(1).tile(1, pred_samples.shape[1], 1)
    mean_scaler = mean_scaler.unsqueeze(1).tile(1, pred_samples.shape[1], 1)
    pred_samples = pred_samples*std_scaler + mean_scaler

    mse = ((pred_samples.mean(axis = 1) - y)**2).mean(axis = 1).mean(axis = 0)
    mae = (abs(pred_samples.median(axis = 1).values - y)).mean()

    crps = torch.zeros((y.shape[0], y.shape[1]))
    for i_var in range(y.shape[1]):
        ED = (abs(pred_samples[:, :, i_var] - y[:, i_var].unsqueeze(1).tile(1, n_samples))).mean(axis = 1)
        EI = (abs(pred_samples[:,:int(n_samples/2), i_var] - pred_samples[:,int(n_samples/2):, i_var])).mean(axis = 1)
        crps[:, i_var] = ED - 0.5*EI
    
    ED = torch.linalg.norm(pred_samples - y.unsqueeze(1).tile(1, n_samples, 1), dim = 2).mean(axis = 1)
    EI = torch.linalg.norm(pred_samples[:, :int(n_samples/2), :] - pred_samples[:, int(n_samples/2):, :], dim = 2).mean(axis = 1)
    es = ED - 0.5*EI

    diff = torch.sqrt(abs(y[:, 0].cpu().detach() - y[:, 1].cpu().detach()))
    VSp = (2*(diff - torch.sqrt(abs(pred_samples[:, :, 0] - pred_samples[:, :, 1])).mean(dim = 1))**2)

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