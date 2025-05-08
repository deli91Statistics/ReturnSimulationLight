import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats

from dataclasses import dataclass
from scipy.optimize import minimize



#%% Error Distribution


@dataclass
class GaussianHyperParam():
    
    n_param = 0
    
    param_bounds = []
    param_grid = []
    

class Gaussian():
    
    def __init__(self):
        self._name = 'gaussian'
        self.est_param = GaussianHyperParam()


    @staticmethod
    def log_density(data, loc, scale):
        return stats.norm.logpdf(data, loc=loc, scale=scale)
    
    
    def create_param_grid(self):
        
        grid = self.est_param.param_grid
        
        return grid
    


@dataclass
class TStudentHyperParam():
    
    n_param = 1     # degree of freedom
    
    param_bounds = [(1, np.inf)]
    param_grid = list(range(1,15))  # degree of freedom
    
    
    
class TStudent():
    
    def __init__(self):
        self._name = 't-student'
        self.est_param = TStudentHyperParam()


    @staticmethod
    def log_density(data, loc, scale, df):
        return stats.t.logpdf(data, loc=loc, scale=scale, df=df)


    def create_param_grid(self):
        
        grid = self.est_param.param_grid
        
        return grid


#%%

class UnivariateReturnModel():
    
    def __init__(self,
                 data,
                 MeanModel = None,
                 VarianceModel = None,
                 ErrorDistribution = None) -> None:
        
        self.data = data
        self.MeanModel = MeanModel
        self.VarianceModel = VarianceModel(self.data)
        self.ErrorDistribution = ErrorDistribution
        self.opt_mean_param = None
        self.opt_var_param = None
        self.opt_dist_param = None
        self.pred_variance = None
        self.std_residuals = None
        
    
    def _split_param(self, params: list):
        
        # n_mean_params = self.MeanModel.n_param
        n_mean_params = 0
        n_var_params = self.VarianceModel.est_param.n_param
        n_dist_params = self.ErrorDistribution.est_param.n_param
        
        mean_params = params[:n_mean_params]
        var_params = params[n_mean_params:n_mean_params + n_var_params]
        dist_params = params[n_mean_params + n_var_params: n_mean_params + n_var_params + n_dist_params]
        
        return mean_params, var_params, dist_params
    
    
    def _manage_llh_fct_input(self, params):
        
        mean_params, var_params, dist_params = self._split_param(params)
        
        loc = 0 # to be implemented
        variance = self.VarianceModel.variance_recursion(data = self.data,
                                                      params = var_params)
        scale = np.sqrt(variance)
        
        dist = dist_params
        
        param_input = {}
        
        if self.ErrorDistribution._name == 'gaussian':
            param_input['loc'] = loc
            param_input['scale'] = scale
        elif self.ErrorDistribution._name == 't-student':
            param_input['loc'] = loc
            param_input['scale'] = scale
            param_input['df'] = dist[0]
        
        return param_input


    def _create_param_grid(self):
        """
        Combine all parameters into a grid
        
        ORDER MATTERS:
        mean params - variance params - distribution params

        At the moment no mean params are considered, need to be implemented

        """
        
        mean_param_grid = ...       # to be implemented
        variance_param_grid = self.VarianceModel.create_param_grid()
        dist_param_grid = self.ErrorDistribution.create_param_grid()
        
        
        if not dist_param_grid:
            param_grid = variance_param_grid
        else:
            param_grid = [np.append(array, number) 
                          for array in variance_param_grid 
                          for number in dist_param_grid]

        return param_grid
    
    
    def neg_log_llh(self, params:list):
        
        params_dict = self._manage_llh_fct_input(params = params)
        
        llh = self.ErrorDistribution.log_density(data = self.data, **params_dict)
        
        return -np.sum(llh)
    
    
    def _grid_search(self):
        
        param_grid = self._create_param_grid()
        
        neg_llh_list = []
        
        for param in param_grid:
            neg_llh = self.neg_log_llh(params = param)
            neg_llh_list.append(neg_llh)

        return param_grid[np.argmin(neg_llh_list)]
    
    
    def _construct_est_param_bounds(self):
        
        mean_bounds = []  # needs to be implemented
        var_bounds = self.VarianceModel.est_param.param_bounds
        dist_bounds = self.ErrorDistribution.est_param.param_bounds
        
        estimation_bounds = mean_bounds + var_bounds + dist_bounds
        
        return estimation_bounds
    
    
    def _construct_est_constraints(self):
        pass
    
    
    def fit(self, grid_search = True, init_param = None):
        
        if grid_search:
            init_param = self._grid_search()
        else:
            init_param = init_param
        
        param_bounds = self._construct_est_param_bounds()
        
        opt = minimize(self.neg_log_llh, 
                       init_param, 
                       bounds=param_bounds,
                       constraints=self.VarianceModel.est_param.param_constraints,
                       method="SLSQP", 
                       tol= None,
                       options=None)
        
        
        opt_mean_param, opt_var_param, opt_dist_param = self._split_param(opt.x)
        
        self.opt_mean_param = opt_mean_param
        self.opt_var_param = opt_var_param
        self.opt_dist_param = opt_dist_param
                     
        return opt.x


    def predicted_mean(self):
        raise NotImplementedError


    def predicted_variance(self) -> np.array:
        """
        Returns the (predicted) variance based on optimized conditional
        variance parameters.

        Returns:
            np.array: Predicted variance
        """
        
        self.pred_variance = self.VarianceModel.variance_recursion(data = self.data,
                                                            params = self.opt_var_param)
        
        return self.pred_variance
    
    
    def standardized_residuals(self) -> np.array:
        
        sigma2 = self.predicted_variance()
        sigma = np.sqrt(sigma2)
        
        self.std_residuals = self.data/sigma
        
        return self.std_residuals