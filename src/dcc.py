import itertools
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm, inv

from typing import List, Tuple, Union
from dataclasses import dataclass

from functions.distributions import ErrorDistribution


@dataclass
class DCCEstHyperParam():
    
    # Hyperparameters for grid search
    arch_grid_bd = [0, 0.3]
    garch_grid_bd = [0.5, 0.99]
    grid_resolution = 10
    
    # need this because dcc grid is generated differently
    def constraint_stationarity(lambda_1: float, lambda_2: float) -> bool:
        return lambda_1 + lambda_2 < 1
    
    
    # Hyperparameters for minimize routine
    param_bounds = [(0,1), (0,1)]
    param_constraints = {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}



class DCC:
    
    def __init__(self,
                 data: Union[pd.DataFrame, np.array],
                 ErrorDistribution: ErrorDistribution):
        
        self.data = data
        self.ErrorDistribution = ErrorDistribution
        self.est_param = DCCEstHyperParam
        self.opt_param = None
        self.pred_correlation = None
    
    
    @staticmethod
    def dcc_11_recursion(data: Union[pd.DataFrame, np.array],
                         lambda_1: float, lambda_2: float, 
                         ) -> np.array:
        """
        Compute the conditional correlation according to Engle(2002). In this case
        specifically, DCC(1,1), therefore only two parameters are required.
        
        The auto-regressive nature of correlation matrices are given by
        
        $$
        Q_t = (1-\lambda_1 - \lambda_2)R 
                + \lambda_1\tilde{\epsilon}_{t-1}\tilde{\epsilon}^{T}_{t-1} 
                + \lambda_2Q_{t-1}
        $$
        
        Check Engle(2002) for more details or obsidian notes

        Args:
            data (Union[pd.DataFrame, np.array]): dataset, in my research, I apply this
                                                  on standardized residuals to filter
                                                  cross-correlation after the univariate
                                                  series were filtered with a GARCH(1,1)
                                                  model.
            lambda_1 (float): "arch" component
            lambda_2 (float): "garch" component

        Returns:
            np.array: Conditional correlation, i.e. series of correlation matrices
            
        Example:
        
        # Case 1: Independent data
        series_1 = np.random.normal(size=1000)
        series_2 = np.random.normal(size=1000)
        series_3 = np.random.normal(size=1000)

        ind_residuals = np.column_stack([series_1, series_2, series_3])

        dcc_11_recursion(data = ind_residuals, lambda_1 = 0.05, lambda_2 = 0.8)


        # Case 2: Dependent data
        mean = [0, 0, 0] 
        cov = [[1, 0.5, 0.3],
               [0.5, 1, 0.2],
               [0.3, 0.2, 1]]

        dep_residuals = np.random.multivariate_normal(mean, cov, 1000)

        dcc_11_recursion(data = dep_residuals, lambda_1 = 0.05, lambda_2 = 0.8)
        
        """
        
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        n_assets = data.shape[1]   
        T_obs = data.shape[0]
        
        # initial value
        R = np.corrcoef(data, rowvar=False)
        Q = R.copy()
        
        # Preallocation of matrix
        R_t = np.zeros((T_obs, n_assets, n_assets))
        
        # DCC recursion
        for t in range(T_obs):
            # R_t[t] = np.diag(np.diag(Q) ** -0.5) @ Q @ np.diag(np.diag(Q) ** -0.5)
            D = np.diag(np.diag(Q))
            D_sqrt_inv = inv(sqrtm(D))
            R_t[t] = D_sqrt_inv @ Q @ D_sqrt_inv
        
            Q = ((1 - lambda_1 - lambda_2) * R 
                 + lambda_1 * np.outer(data[t, :], data[t, :]) 
                 + lambda_2 * Q)
                
        return R_t
    
    
    def neg_likelihood_fct(self, params: list) -> float:
        """
        Computes the negative log-likelihood function given the
        dataset. Conditional correlation based on some parameters
        are computed first and then the multivariate log-density
        is evaluated given each time point. The parameters are later
        optimized in a maximum-likelihood style.

        Args:
            params (list): dcc recursion parameters, contains only
                           two parameters since a DCC(1,1) model is
                           estimated.

        Returns:
            float: negative log-likelihood
        """
        
        R_t = self.dcc_11_recursion(self.data,
                                    lambda_1 = params[0],
                                    lambda_2 = params[1])
        
        # maybe provide inverted matrix already, so it does not invert 
        # during optimization?
        
        # need this check otherwise list comprehension down belowdoesn'T work, 
        # col index is returned instead of the row with data
        if isinstance(self.data, pd.DataFrame):
            data = self.data.values
        else: 
            data = self.data
        
        lls = [self.ErrorDistribution.multivariate_log_density(row, C) 
               for row, C in zip(data, R_t)]
        
        return -np.sum(lls)
    
    
    def _create_dcc_param_grid(self) -> List[Tuple[float, ...]]:
        """
        Given the range where the parameters should be located. Create
        a grid and derive all combinations. Filter all pairs where
        the sum of the two parameters is larger than 1 for stationarity
        purposes.

        Returns:
            List[Tuple[float, ...]]: Array of parameter pairs
        """

        # Create interval spacings for the given parameter bounds
        param_ranges = [np.linspace(start, end, num=self.est_param.grid_resolution)
                        for start, end in [self.est_param.arch_grid_bd,
                                           self.est_param.garch_grid_bd]]

        param_grid = list(itertools.product(*param_ranges))

        # Filter pairs that do not match stationary requirement
        param_grid = [params for params in param_grid 
                      if self.est_param.constraint_stationarity(lambda_1 = params[0],
                                                             lambda_2 = params[1])]

        return param_grid
    
    
    def _search_param_grid(self) -> np.array:
        """
        Create a grid of parameters. Compute negative log-likelihood
        function for each possible parameter set and select then minimum
        out of all combinations. Return the parameters as starting
        values for minimization routine.

        Returns:
            np.array: Array of "optimal" starting values
        """
        
        param_grid = self._create_dcc_param_grid()
        
        neg_llhs = []
        
        for param in param_grid:
            neg_llh = self.neg_likelihood_fct(params = param)
            neg_llhs.append(neg_llh)
            
        return param_grid[np.argmin(neg_llhs)]


    def fit(self, grid_search = True, init_param = None):
        """
        Fit the model and save optimal parameters and predicted correlation
        
        """
    
        if grid_search:
            init_param = self._search_param_grid()

        opt = minimize(self.neg_likelihood_fct, 
                       init_param, 
                       bounds = self.est_param.param_bounds,
                       constraints = self.est_param.param_constraints)
        
        self.opt_param = opt.x
        self.predicted_correlation()
            
    
    def predicted_correlation(self) -> np.ndarray:
        """
        Compute the in-sample predicted correlation based on optimal parameters

        Returns:
            np.ndarray: Series of predicted correlation coefficients.
        """
        
        pred_correlation = self.dcc_11_recursion(data = self.data,
                                                 lambda_1 = self.opt_param[0],
                                                 lambda_2 = self.opt_param[1])
        
        self.pred_correlation = pred_correlation
        
        return self.pred_correlation
    

    def _std_cross_corr_filtered_resid(self, 
                                       asset_returns: pd.DataFrame,
                                       asset_variances: pd.DataFrame) -> np.array:
        """
        Filter data for auto-cross correlation. See Engle (2002) or
        STATA documention for model description.
        
        The prdicted variance of each single assets must be provided
        beforehand and won't be computed by the DCC class.
        
        NEED TO RETHINK THE STRUCTURE OF DCC IN GENERAL IF ASSET RETURNS
        ARE NOT INSTANCE VARIABLES!!!!!!!!!!!!

        Returns:
            np.array: Series of auto-cross correlation filtered residuals
        """
        
        if isinstance(asset_returns, pd.DataFrame):
            asset_returns_val = asset_returns.values
        else:
            asset_returns_val = asset_returns
        
        
        if isinstance(asset_variances, pd.DataFrame):
            asset_variances = asset_variances.values
        
        
        t = asset_returns_val.shape[0]
        
        D_t_sqrt = [np.diag(row) for row in np.sqrt(asset_variances)]
        H_t = D_t_sqrt @ self.pred_correlation @ D_t_sqrt
        
        H_t_chol = np.linalg.cholesky(H_t)
        inv_H_t_chol = np.linalg.inv(H_t_chol)          # maybe not so efficient
        
        filtered_resid = np.array([inv_H_t_chol[i] @ asset_returns_val[i] 
                                   for i in range(t)])
        
        if isinstance(asset_returns, pd.DataFrame):
            filtered_resid = pd.DataFrame(columns=asset_returns.columns,
                                          index = asset_returns.index,
                                          data = filtered_resid)
        
        return filtered_resid
    
    
    @staticmethod
    def simulate(lambda_1: float, lambda_2: float, 
                 std_residuals: np.ndarray, dcc_init_R: np.array = None):
        
        if isinstance(std_residuals, pd.DataFrame):
            std_residuals = std_residuals.values
        
        if dcc_init_R is not None:
            R = dcc_init_R
        else:
            R = np.corrcoef(std_residuals, rowvar=False)
        
        n_asset = R.shape[0]
        n_sim = std_residuals.shape[0]
        
        Q = R.copy()
        
        R_t = np.zeros((n_sim, n_asset, n_asset))

        for idx in range(n_sim):
            R_t[idx] = np.diag(np.diag(Q) ** -0.5) @ Q @ np.diag(np.diag(Q) ** -0.5)
            Q = ((1 - lambda_1 - lambda_2) * R 
                 + lambda_1 * np.outer(std_residuals[idx, :], std_residuals[idx, :]) 
                 + lambda_2 * Q)
        
        return R_t