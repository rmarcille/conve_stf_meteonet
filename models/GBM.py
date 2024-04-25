from sklearn.ensemble import GradientBoostingRegressor
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import torch
import numpy as np
import random
import time

class GBM_quantile:
    def __init__(self, 
                 n_previ, 
                 params,
                 X_base, 
                 y, 
                 lr, 
                 n_est, 
                 max_depth,
                 min_samples_leaf=20,
                 min_samples_split=20):
        self.X_features_labels = X_base.features_labels
        self.params = params
        self.n_param = len(params)
        self.n_previ = n_previ
        self.alphas = [0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95]
        self.n_alphas = len(self.alphas)
        self.models = {}
        self.phases = ['train', 'val', 'test']
        for param in self.params:
            self.models[param] = [{} for th in range(n_previ)]
        h_params = dict(learning_rate=lr,
                        n_estimators=n_est,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=min_samples_split)
        for param in self.params:
            for th in range(self.n_previ):
                for alpha in self.alphas:
                    self.models[param][th][alpha] = GradientBoostingRegressor(loss="quantile", alpha=alpha, **h_params)

    def fit_time_lead(self, time_lead):
        models = {}
        for alpha in tqdm(self.alphas):
            models[alpha] = {}
            for i, param in enumerate(self.params):
                models[alpha][param] = self.models[self.params[i]][time_lead][alpha]
                models[alpha][param].fit(self.X['train'][:, :, time_lead], self.y['train'][:, i, time_lead])
        return models

    def fit_single(self, time_lead, alpha, param):
        t0 = time.time()
        self.models[self.params[param]][time_lead][self.alphas[alpha]].fit(self.X['train'][:, :, time_lead], self.y['train'][:, param, time_lead])
        t1 = time.time()
        print(f'GBM  fitting time = {round(t1 - t0, 2)}')
        

    def fit(self, range = range(60)):
        for th in tqdm(range):
            print(f'Fit {th} timestep model')
            t0 = time.time()
            for alpha in tqdm(self.alphas):
                for i, param in enumerate(self.params):
                    self.models[self.params[i]][th][alpha].fit(self.X['train'][:, :, th], self.y['train'][:, i, th])
            t1 = time.time()
            print('Elapsed time : ', round(t1 - t0, 2))
    
    def generate_pred(self, X):
        self.pred = {}

        for phase in self.phases:
            gbm_pred = torch.zeros((X[phase].shape[0], self.n_param, self.n_alphas, self.n_previ)).numpy()
            for p, param in enumerate(self.params):
                for th in range(self.n_previ):
                    for a, alpha in enumerate(self.alphas):
                        gbm_pred[:, p, a, th] = self.models[param][th][alpha].predict(X[phase][:, :, th].numpy())
            self.pred[phase] = gbm_pred
        return self.pred
    
    def generate_samples(self, n_samples = 400, phases = ['test']):
        self.samples = {}
        for phase in phases:
            ypred = torch.tensor(self.pred[phase]).permute(1, 2, 0, 3).contiguous().view(self.n_param, len(self.alphas), -1).permute(2, 0, 1)
            samples = torch.zeros((ypred.shape[0], n_samples, self.n_param))
            with torch.no_grad():
                for t in tqdm(range(ypred.shape[0])):
                    for p, param in enumerate(self.params): 
                        f = CubicSpline(self.alphas, np.sort(ypred[t, p, :]), bc_type='natural')
                        x_in = np.array([f(random.random()) for i in range(n_samples)]).reshape(n_samples)
                        samples[t, :, p] = torch.tensor(x_in)
            self.samples[phase] = samples
        return self.samples

    def generate_samples_time_lead(self, time_lead, phase, n_samples = 1000):
        y = self.y[phase].permute(1, 0, 2).contiguous().view(self.n_param, -1).permute(1, 0)
        ypred = self.pred[phase].permute(1, 2, 0, 3).contiguous().view(self.n_param, len(self.alphas), -1).permute(2, 0, 1)
        samples = torch.zeros((n_samples, y.shape[1]))
        with torch.no_grad():
            for p, param in enumerate(self.params): 
                f = CubicSpline(self.alphas, np.sort(ypred[time_lead, p, :]), bc_type='natural')
                x_in = np.array([f(random.random()) for i in range(n_samples)]).reshape(n_samples)
                samples[:, p] = torch.tensor(x_in)
        return samples

