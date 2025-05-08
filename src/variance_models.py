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
        
    

@dataclass
class GJRGARCHHyperParam():
    
    n_param = 4    # omega, alpha, gamma, beta
    
    # Based on sheppards algorith, the linspace approach doesn't work in that simplicity
    # arch_grid_bd = [0.01, 0.4]
    # garch_grid_bd = [0.5, 0.98]
    # grid_resolution = 10
    
    arch_grid_bd = [0.01, 0.05, 0.1, 0.2, 0.3]
    garch_grid_bd = [0.5, 0.6, 0.7, 0.9, 0.98]
    
    param_bounds = [(0,1), (0,1), (0,1), (0,1)]
    param_constraints = [
        {'type': 'ineq', 'fun': lambda param: -(param[1] + 0.5*param[2] + param[3] - 1)},
        {'type': 'ineq', 'fun': lambda param: param[1]},
        {'type': 'ineq', 'fun': lambda param: param[2]},
        {'type': 'ineq', 'fun': lambda param: param[3]}
        ]

    
    
class GJRGARCH():
    
    def __init__(self, data=None):
          
        self.data = data
        self.est_param = GJRGARCHHyperParam()
       
       
    @staticmethod
    def variance_recursion(data: pd.Series,
                           params: list,
                           init_sigma2: Optional[None] = None) -> np.array:
        
        const = params[0]
        arch_coef = params[1]
        lev_coef = params[2]
        garch_coef = params[3]

        
        T = len(data)
        sigma_2 = np.zeros(T)
    
        if init_sigma2 is None:
            init_sigma2 = np.var(data)
            
        sigma_2[0] = init_sigma2
        
        for i in range(1, T):
            sigma_2[i] = (const + arch_coef * data[i-1]**2 
                            + lev_coef * data[i-1]**2 * (data[i-1]<0) 
                            + garch_coef * sigma_2[i-1])
            
        return sigma_2


    def create_param_grid(self):
        
        # ARCH, GARCH, Leverage component grid
        # alpha = np.linspace(self.est_param.arch_grid_bd[0], 
        # # self.est_param.arch_grid_bd[1], 
        # # num=self.est_param.grid_resolution)
                            
        # beta = np.linspace(self.est_param.garch_grid_bd[0],
        # self.est_param.garch_grid_bd[1], 
        # num=self.est_param.grid_resolution)

        alpha = self.est_param.arch_grid_bd
        gamma = self.est_param.arch_grid_bd
        beta = self.est_param.garch_grid_bd
            
        grid = list(itertools.product(*[alpha, gamma, beta]))       # gamma = alpha

        # Starting value for baseline volatility
        const = np.mean(np.abs(self.data)**2)
        
        # Construct grid with constraints
        # Based on this algorithm, the prespecified grid values must be choosen, i.e.
        # linspace doesn't apply here
        
        lst_starting_parameters = []
        for values in grid:
            sv = np.ones(4)
            alpha, gamma, beta  = values
            sv[0] = const * (1 - beta)                  # omega
            sv[1] = alpha                               # alpha_1
            sv[2] = gamma                               # gamma_1
            sv[3] = beta - alpha - 0.5*gamma            # beta_1
            
            lst_starting_parameters.append(sv)

        return lst_starting_parameters
    
    
    @staticmethod    
    def simulate(params: list, 
                 sigma2_0: float = None, 
                 innovations: np.array = None):

        const = params[0]
        arch_coef = params[1]
        lev_coef = params[2]
        garch_coef = params[3]
          
        n_sim = len(innovations)
        
        eps = innovations
              
        resid = np.zeros(n_sim)
        sigma2 = np.zeros(n_sim)


        if sigma2_0 is None:
            sigma2[0] = const / (1 - arch_coef - 0.5*lev_coef - garch_coef)
        else:
            sigma2[0] = sigma2_0
        
        for i in range(1, n_sim):
            sigma2[i] = (const + arch_coef*resid[i-1]**2 
                               + lev_coef*resid[i-1]**2*(resid[i-1]<0) 
                               + garch_coef*sigma2[i-1])
            resid[i] = np.sqrt(sigma2[i]) * eps[i]
            
        return resid, sigma2  