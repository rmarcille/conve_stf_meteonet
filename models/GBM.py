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
                 min_samples_leaf = 9,
                 min_samples_split = 9):
        self.X_features_labels = X_base.features_labels
        self.X = X_base.X
        self.y = y
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
                    self.models[param][th][alpha] = GradientBoostingRegressor(loss="quantile", 
                                                                              alpha=alpha, 
                                                                              **h_params)

    def fit(self):        
        for th in tqdm(range(self.n_previ)):
            print(f'Fit {th} timestep model')
            t0 = time.time()
            for alpha in self.alphas:
                self.models[self.params[0]][th][alpha].fit(self.X['train'][:, :, th], self.y['train'][:, 0, th])
                self.models[self.params[1]][th][alpha].fit(self.X['train'][:, :, th], self.y['train'][:, 1, th])
            t1 = time.time()
            print('Elapsed time : ', round(t1 - t0, 2))
    
    def pred(self):
        self.pred = {}

        for phase in self.phases:
            gbm_pred = torch.zeros((self.X[phase].shape[0], self.n_param, self.n_alphas, self.n_previ)).numpy()
            for p, param in enumerate(self.params):
                for th in range(self.n_previ):
                    for a, alpha in enumerate(self.alphas):
                        gbm_pred[:, p, a, th] = self.models[param][th][alpha].predict(self.X[phase][:, :, th].numpy())
            self.pred[phase] = gbm_pred
    

    def generate_samples(self, n_samples = 400):
        self.samples = {}
        for phase in self.phases:
            y = self.y[phase].permute(1, 0, 2).contiguous().view(self.n_param, -1).permute(1, 0)
            ypred = self.pred[phase].permute(1, 2, 0, 3).contiguous().view(self.n_param, 21, -1).permute(2, 0, 1)
            samples = torch.zeros((y.shape[0], n_samples, self.n_param))
            with torch.no_grad():
                for p, param in enumerate(self.params):
                    samples_param = torch.zeros((y.shape[0], n_samples))
                    for t in tqdm(range(y.shape[0])):
                        f = CubicSpline(self.alphas, ypred[t, 0, :], bc_type='natural')
                        x_in = np.array([f(random.random()) for i in range(n_samples)]).reshape(n_samples)
                        samples_param[t, :] = torch.tensor(x_in)
                        samples[t, :, p] = samples_param[t, :]
            self.samples[phase] = samples
        return self.samples

