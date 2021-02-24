# Written by Alexander Kerr

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

class Dist(object):
    """
    """
    
    def __init__(self, X, metric):
        self.dist = pdist(X, metric=metric)
        self.min_dist = np.min(self.dist)
        self.max_dist = np.max(self.dist)
    
class SimMat(Dist):
    """
    """
    
    def __init__(self, X, metric):
        super().__init__(X, metric)
        
    def __call__(self, eps):
        return self.solve(eps)
        
    @staticmethod
    def gauss(dist, eps):
        return np.exp(-dist*dist/eps)
    
    @staticmethod
    def square(dist1d):
        return squareform(dist1d)
    
    def solve(self, eps):
        return self.__solve(eps)
        
    def __solve(self, eps):
        return self.gauss(self.square(self.dist), eps)
    
class DMap(SimMat):
    """
    The diffusion map operator.
    
    Parameters
    ----------
    X : array-like
        The data of which diffusions distances are determined.
    metric : str, or function
        Distance metric (see scipy.spatial.distance)
        
    Keywords
    --------
    n_dims : int
        Number of diffusion modes to return.  Defaults to 10.
    """
    
    def __init__(self, X, metric, n_modes=10):
        super().__init__(X, metric)
        self.n_modes = n_modes
        
    def __call__(self, eps, t=500):
        return self.solve(eps, t=t)
        
    def solve(self, eps, t=500):
        return self.__solve(eps, t=t)
                
    def __solve(self, eps, t=500):
        """
        Return the diffusion modes w/ eigenvalues
        
        Parameters
        ----------
        ep : float
            Resolution parameter.  Length-scale determining similarity.
            
        Keywords
        --------
        Diffusion time.  Defaults to 500.
        """
        
        sim_mat = self._SimMat__solve(eps)
        prob_mat = np.matmul(np.diag(1/np.sum(sim_mat, axis=1)), sim_mat)
        
        val, vec = np.linalg.eig(prob_mat)
        
        sorted_id = np.argsort(val.real)[:-self.n_modes-1:-1]
        
        self.dval = val[sorted_id].real**t
        self.dmode = vec[:,sorted_id].real*val[None,sorted_id].real
        
        return self.dval, self.dmode
    
class DCluster(DMap):
    """
    KMeans clustering in diffusion space.
    """
    
    def __init__(self, X, metric, n_clusters, n_modes=10):
        super().__init__(X, metric, n_modes=n_modes)
        self.n_clusters = n_clusters
        
    def __call__(self, eps, t=500):
        return self.solve(eps, t=t)
        
    def solve(self, eps, t=500):
        return self.__solve(eps, t=t)
        
    def __solve(self, eps, t=500):
        self._DMap__solve(eps, t=t)
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.labels_ = self.kmeans.fit_predict(self.dmode)
        return self.labels_