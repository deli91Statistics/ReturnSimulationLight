import numpy as np
import pandas as pd
import itertools

from typing import Optional
from dataclasses import dataclass



@dataclass
class GARCHHyperParam():
    
    n_param = 3 # omega, alpha, beta
    
    # Hyperparameters for grid search
    arch_grid_bd = [0.001, 0.4]
    garch_grid_bd = [0.5, 0.98]
    grid_resolution = 10
    
    # Hypterparameters for minimize routine
    param_bounds = [(0,1), (0,1), (0,1)]
    param_constraints = [{'type': 'ineq', 'fun': lambda param: -(param[1]+param[2]-1)},
                         {'type': 'ineq', 'fun': lambda param: param[1]},
                         {'type': 'ineq', 'fun': lambda param: param[2]}]
    

class GARCH():

    def __init__(self, data = None):
        self.data = data
        self.est_param = GARCHHyperParam()


    @staticmethod
    def variance_recursion(data: pd.Series,
                           params: list, 
                           init_sigma2: Optional[float] = None) -> np.array:
 
        const = params[0]
        arch_coef = params[1]
        garch_coef = params[2]
        
        T = len(data)
        sigma_2 = np.zeros(T)
        
        if init_sigma2 is None:
            init_sigma2 = np.var(data)
            
        sigma_2[0] = init_sigma2

        for i in range(1, T):
                sigma_2[i] = (const 
                              + arch_coef * data[i-1]**2 
                              + garch_coef * sigma_2[i-1])
        
        return sigma_2
    
    
    def create_param_grid(self) -> list:

        # Create ARCH and GARCH grid
        arch_grid = np.linspace(self.est_param.arch_grid_bd[0], 
                                self.est_param.arch_grid_bd[1], 
                                num=self.est_param.grid_resolution)
        
        garch_grid = np.linspace(self.est_param.garch_grid_bd[0], 
                                 self.est_param.garch_grid_bd[1], 
                                 num=self.est_param.grid_resolution)

        grid = list(itertools.product(*[arch_grid, garch_grid]))

        # Starting value for baseline volatility omega
        const = np.mean(abs(self.data) ** 2)


        # Construct grid with constraints
        lst_starting_parameters = []
        
        for values in grid:
            sv = np.ones(3)
            pre_arch, pre_garch = values
            sv[0] = const * (1 - pre_garch)     # omega
            sv[1] = pre_arch                    # alpha_1
            sv[2] = pre_garch - pre_arch        # beta_1
            
            lst_starting_parameters.append(sv)

        return lst_starting_parameters
    
    
    @staticmethod    
    def simulate(params: list, 
                 sigma2_0: float = None, 
                 innovations: np.array = None):

        const = params[0]
        arch_coef = params[1]
        garch_coef = params[2]
          
        n_sim = len(innovations)
        
        eps = innovations
              
        resid = np.zeros(n_sim)
        sigma2 = np.zeros(n_sim)


        if sigma2_0 is None:
            sigma2[0] = const / (1 - arch_coef - garch_coef)
        else:
            sigma2[0] = sigma2_0
        
        for i in range(1, n_sim):
            sigma2[i] = const + arch_coef * (resid[i - 1] ** 2) + garch_coef * sigma2[i - 1]
            resid[i] = np.sqrt(sigma2[i]) * eps[i]
            
        return resid, sigma2
        
    
