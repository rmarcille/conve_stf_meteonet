import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import pinv
import torch 
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# from scoring.scores import proba_scores



class Analogs():
    def __init__(self, 
                 X, 
                 target, 
                 k, 
                 n_pcs = 20, 
                 regression = 'locally-constant',
                 leaf_size = 50,
                 distance = 'euclidean'):
        self.k = k
        
        #global approach
        self.neighborhood = np.ones([k, k])
        self.nt_forecast = target['train'].shape[-1]

        self.regression = regression
        self.n_pcs = n_pcs
        self.leaf_size = leaf_size
        self.distance = distance

        #Catalog is a PCA of input of shape (vars, lead time)
        # X_catalog = torch.cat((X['train'], X['val']), dim = 0)
        X_catalog = X['train']

        self.catalog = self.analogs_compute_PCA(X_catalog.view(X_catalog.shape[0], -1))
        # self.successors = torch.concat((target['train'], target['val']), dim = 0).permute(0, 2, 1)
        self.successors = target['train'].permute(0, 2, 1)

    def analogs_compute_PCA(self, X):
        lim = 0.995
        self.sc = StandardScaler()
        X_scaled = self.sc.fit_transform(X)
        self.pca = PCA(n_components = lim)
        self.pca.fit(X_scaled)
        PCs = self.pca.transform(X_scaled)
        catalog_PCs = torch.tensor(PCs[:,:self.n_pcs])
        return catalog_PCs

    def analogs_input_transform(self, X):
        Xtest = self.sc.transform(X)
        PCs_Xtest_base = self.pca.transform(Xtest)
        Xtest_pca = torch.tensor(PCs_Xtest_base[:,:self.n_pcs])

        return Xtest_pca

    def AnDA_analog_forecasting(self, x, n_samples = 1000):
        N = x.shape[0]
        n = self.successors.shape[-1]
        xf = np.zeros([n_samples, N, self.nt_forecast, n])
        xf_mean = np.zeros([N, self.nt_forecast, n])
        xf_cov = np.zeros([N, self.nt_forecast, n, n])
        # stop_condition = 0
        i_var = np.array([0])

        if np.all(self.neighborhood == 1):
            i_var = np.arange(n, dtype=np.int64)
            # stop_condition = 1

        search_catalog = self.catalog
        print('Catalog shape = ', self.catalog.shape)
        print('Search catalog shape = ', search_catalog.shape)

        for neighbor in range(self.k):
            kdt = KDTree(search_catalog, leaf_size=self.leaf_size, metric=self.distance)
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
                if self.regression == "locally_constant":
                
                    # compute the analog forecasts
                    xf_tmp[:, i_var] = self.successors[:, th, :][np.ix_(index_knn[i_N, :], i_var)]
                    
                    # weighted mean and covariance
                    xf_mean[i_N, th, i_var] = np.sum(xf_tmp[:, i_var]*np.repeat(weights[i_N, :][np.newaxis].T, len(i_var), 1), 0)
                    E_xf = (xf_tmp[:, i_var] - np.repeat(xf_mean[i_N, th, i_var][np.newaxis], self.k, 0)).T
                    cov_xf = 1.0/(1.0-np.sum(np.power(weights[i_N, :],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T)
                # method "locally-linear"
                elif (self.regression == 'local_linear'):
            
                    # define analogs, successors and weights
                    X = np.array(search_catalog[np.ix_(index_knn[i_N,:], i_var)])          
                    Y = np.array(self.successors[:, th, :][np.ix_(index_knn[i_N,:],i_var)])                
                    w = weights[i_N,:][np.newaxis]
                    
                    # compute centered weighted mean and weighted covariance
                    Xm = np.sum(X*w.T, axis=0)[np.newaxis]
                    Xc = X - Xm
                    
                    # regression on principal components
                    Xr   = np.c_[np.ones(X.shape[0]), Xc]
                    Cxx  = np.dot(w    * Xr.T,Xr)
                    Cxx2 = np.dot(w**2 * Xr.T,Xr)
                    Cxy  = np.dot(w    * Y.T, Xr)
                    inv_Cxx = pinv(Cxx, rcond=0.01) # in case of error here, increase the number of analogs (AF.k option)
                    beta = np.dot(inv_Cxx,Cxy.T)
                    X0 = x[i_N,i_var]-Xm
                    X0r = np.c_[np.ones(X0.shape[0]),X0]
                    
                    # weighted mean
                    xf_mean[i_N, th, i_var] = np.dot(X0r,beta)
                    pred = np.dot(Xr,beta)
                    res = Y-pred
                    xf_tmp[:,i_var] = xf_mean[i_N, th, i_var][np.newaxis] + res
        
                    # weigthed covariance
                    cov_xfc = np.dot(w * res.T,res)/(1-np.trace(np.dot(Cxx2,inv_Cxx)))
                    cov_xf = cov_xfc*(1+np.trace(Cxx2@inv_Cxx@X0r.T@X0r@inv_Cxx))
                    
                    # constant weights for local linear
                    weights[i_N,:] = 1.0/len(weights[i_N,:])
                    
                xf[:, i_N, th, i_var] = np.random.multivariate_normal(xf_mean[i_N, th, i_var], cov_xf, n_samples)
                xf_cov[i_N, th, :, :] = cov_xf
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
        
