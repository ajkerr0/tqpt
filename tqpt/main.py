# Written by Alexander Kerr

from itertools import product

import numpy as np

from .diffmap import DCluster
    
class TQPTBase(DCluster):
    """
    """
    
    def __init__(self, X_func, param, metric, n_clusters, n_modes=10):
        xf, X = X_func(*param)
        super().__init__(X, metric, n_clusters, n_modes=n_modes)
        self.xf = xf
        self.n_dim = len(param)
        
    def solve(self, eps, t=500):
        return self.__solve(eps, t=t)
        
    def __solve(self, eps, t=500):
        self._DCluster__solve(eps, t=t)
        bid = np.where(np.abs(np.diff(self.labels_)) > 0)[0]
        self.boundary = (self.xf[bid] + self.xf[bid+1])/2.
        return self.boundary
        
class TQPT(TQPTBase):
    """
    Core class of tqpt
    """
    
    def __init__(self, X_func, param, metric, n_clusters, n_modes=10):
        self.X_func = X_func
        self.param = param
        self.metric = metric
        self.n_clusters = n_clusters
        self.n_modes = n_modes
        
    def solve(self, eps, t=500):
        return self.__solve(eps, t=t)
        
    def __solve(self, eps, t=500):
        diagram_data = []
        for combo in list(product(*self.param)):
            super().__init__(self.X_func, combo, self.metric, self.n_clusters, 
                             n_modes=self.n_modes)
            self._TQPTBase__solve(eps, t=t)
            for boundary in self.boundary:
                loc = list(combo)
                loc.append(boundary)
                diagram_data.append(loc)
        self.diagram_data = np.array(diagram_data)
            
            
        
        
        
    
        
        
        
        
    
    