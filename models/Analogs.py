import numpy as np
from sklearn.neighbors import KDTree
import torch 
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

class Analogs():
    def __init__(self, X, target, k, regression = 'locally-constant'):
        self.k = k
        self.neighborhood = np.ones([k, k])
        self.nt_forecast = target['train'].shape[-1]

        #Catalog is a PCA of input of shape (vars, lead time)
        X_catalog = torch.cat((X['train'], X['val']), dim = 0)
        self.catalog = self.analogs_compute_PCA(X_catalog.view(X_catalog.shape[0], -1), n_pcs = 20)
        self.successors = torch.concat((target['train'], target['val']), dim = 0).permute(0, 2, 1)
        self.regression = regression

    def analogs_compute_PCA(self, X, n_pcs = 20):
        lim = 0.995
        self.sc = StandardScaler()
        X_scaled = self.sc.fit_transform(X)
        self.pca = PCA(n_components = lim)
        self.pca.fit(X_scaled)
        PCs = self.pca.transform(X_scaled)
        catalog_PCs = torch.tensor(PCs[:,:n_pcs])
        return catalog_PCs

    def analogs_input_transform(self, X, n_pcs = 20):
        Xtest = self.sc.fit_transform(X)
        PCs_Xtest_base = self.pca.transform(Xtest)
        Xtest_pca = torch.tensor(PCs_Xtest_base[:,:n_pcs])

        return Xtest_pca

    def AnDA_analog_forecasting(self, x, n_samples = 1000):
        N = x.shape[0]
        n = self.successors.shape[-1]
        xf = np.zeros([n_samples, N, self.nt_forecast, n])
        xf_mean = np.zeros([N, self.nt_forecast, n])
        xf_cov = np.zeros([N, self.nt_forecast, n, n])
        stop_condition = 0
        i_var = np.array([0])

        if np.all(self.neighborhood == 1):
            i_var = np.arange(n, dtype=np.int64)
            stop_condition = 1

        search_catalog = self.catalog
        print('Catalog shape = ', self.catalog.shape)
        print('Search catalog shape = ', search_catalog.shape)

        for neighbor in range(self.k):
            kdt = KDTree(search_catalog, leaf_size=50, metric='euclidean')
            dist_knn, index_knn = kdt.query(x, self.k)
        lambdaa = np.median(dist_knn)

        # compute weights
        if self.k == 1:
            weights = np.ones([N,1])
        else:
            weights = self.mk_stochastic(np.exp(-np.power(dist_knn,2)/lambdaa**2))

        for i_N in range(0, N):
            for th in range(self.nt_forecast):
                # initialization
                xf_tmp = np.zeros([self.k, np.max(i_var)+1])
            
                # compute the analog forecasts
                xf_tmp[:, i_var] = self.successors[:, th, :][np.ix_(index_knn[i_N, :], i_var)]
                
                # weighted mean and covariance
                xf_mean[i_N, th, i_var] = np.sum(xf_tmp[:, i_var]*np.repeat(weights[i_N, :][np.newaxis].T, len(i_var), 1), 0)
                E_xf = (xf_tmp[:, i_var] - np.repeat(xf_mean[i_N, th, i_var][np.newaxis], self.k, 0)).T
                cov_xf = 1.0/(1.0-np.sum(np.power(weights[i_N, :],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T)
                xf[:, i_N, th, i_var] = np.random.multivariate_normal(xf_mean[i_N, th, i_var], cov_xf, n_samples)
                xf_cov[i_N, th, :, :] = cov_xf

        if (np.array_equal(i_var,np.array([n-1])) or (len(i_var) == n)):
            stop_condition = 1
                
        else:
            i_var = i_var + 1

        return xf, xf_mean, xf_cov
    
    def pred(self, xtest, n_samples = 200):
        mean_pred, samples_pred, cov_pred = self.AnDA_analog_forecasting(xtest, n_samples)
        return mean_pred, samples_pred, cov_pred

    def normalise(self, M):
        """ Normalize the entries of a multidimensional array sum to 1. """

        c = np.sum(M)
        # Set any zeros to one before dividing
        d = c + 1*(c==0)
        M = M/d
        return M

    def mk_stochastic(self, T):
        """ Ensure the matrix is stochastic, i.e., the sum over the last dimension is 1. """

        if len(T.shape) == 1:
            T = self.normalise(T)
        else:
            n = len(T.shape)
            # Copy the normaliser plane for each i.
            normaliser = np.sum(T,n-1)
            normaliser = np.dstack([normaliser]*T.shape[n-1])[0]
            # Set zeros to 1 before dividing
            # This is valid since normaliser(i) = 0 iff T(i) = 0

            normaliser = normaliser + 1*(normaliser==0)
            T = T/normaliser.astype(float)
        return T

    def sample_discrete(self, prob, r, c):
        """ Sampling from a non-uniform distribution. """

        # this speedup is due to Peter Acklam
        cumprob = np.cumsum(prob)
        n = len(cumprob)
        R = np.random.rand(r,c)
        M = np.zeros([r,c])
        for i in range(0,n-1):
            M = M+1*(R>cumprob[i]);    
        return int(M)
        
