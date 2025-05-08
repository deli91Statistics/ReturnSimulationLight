import numpy as np
import pandas as pd

from scipy.special import gammaln
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import t, multivariate_t, norm
from scipy.optimize import minimize


from typing import Union
from dataclasses import dataclass

from functions.constraints import (constraint_unit_diagonal, 
                                   constraint_is_positive_definite)



@dataclass
class CopulaGaussianHyperParam:
    cons = [{'type': 'eq', 'fun': constraint_unit_diagonal}]
    
    def generate_param_bounds(n_coefs):
        """
        Used to bound the correlation coefficients within 
        -1 and 1
        """
        return [(-1,1)] * n_coefs
    

class CopulaGaussian:
    
    def __init__(self,
                 unif_data: Union[np.ndarray, pd.DataFrame]):
        self.unif_data = unif_data
        self.est_param = CopulaGaussianHyperParam
        self.opt_param = None
        

    def log_density(self, R: np.ndarray) -> float:
        """
        Evaluates the log-density function of the Gaussian copula tailored 
        for optimization purposes. This function omits terms that are constant 
        with respect to the parameters being optimized.    

        Gaussian copula log-density:
        \( \log f(\mathbf{u}; \Sigma)=-\frac{1}{2} \mathbf{z}^T \Sigma^{-1} \mathbf{z} 
        - \frac{1}{2} \log |\Sigma| - \sum_{i=1}^d \log \phi(z_i) \)
        
        where \(\mathbf{z}\) is the vector with elements \(z_i = \Phi^{-1}(u_i)\), 
        and \(\phi\) is the standard normal density function.
        
        Note: In this optimization function, the terms \(\sum_{i=1}^{d} \log \phi(z_{i})
        have been omitted as they are constant with respect to the parameters in \Sigma.
        This function is NOT suitable for model comparison using information criteria.


        Args:
            u (Union[np.ndarray, pd.DataFrame]): A (n x d) matrix (numpy ndarray or 
                                                pandas DataFrame) of copula data, 
                                                where n is the number of samples and 
                                                d is the dimension.
            R (np.ndarray): A (d x d) numpy ndarray correlation matrix.

        Returns:
            float: array of log-density evaluated at u.
        """

        if isinstance(self.unif_data, pd.DataFrame):
            u = self.unif_data.values
        else:
            u = self.unif_data

        z = norm.ppf(u)
        L = cholesky(R, lower=True)
        det_Sigma = np.prod(np.diag(L))**2

        term1 = -0.5 * np.log(det_Sigma)
        inv_Lz = solve_triangular(L, z.T, lower=True)
        term2 = -0.5 * np.sum(inv_Lz**2, axis=0)

        # Sum the terms to get the log-likelihood:
        log_density = term1 + term2
        
        return log_density


    def neg_copula_llh(self, R: np.ndarray) -> float:
        """
        Computes negative log-likelihood function

        Args:
            R (np.ndarray): correlation matrix

        Returns:
            float: neg log likelihood value given some parameters
        """
        return -np.sum(self.log_density(R=R))


    def MLE_objective_fct(self, param: np.array) -> float:
        """
        Implements MLE objective function that is optimized. Input parameter obtained 
        from flattened lower triangular matrix after cholesky decomoposition in order
        to optimize unique elements (correlation matrix is symmetric).
        
        Need to reconstruct correlation matrix from flattened vector before passing
        to log-likelihood/log density function.

        Args:
            param (np.array): Flattened vector with unique correlation coef. and zeros

        Returns:
            float: negative log-likelihood value given parameters
        """
        
        n = self.unif_data.shape[1]
        L = param.reshape(n, n)
        R = L @ L.T
    
        return self.neg_copula_llh(R)
    
    
    def fit(self, init_R: np.array = None) -> None:
        """
        Optimization routine for obtaining maximum likelihood estimation.
        Compute cholesky decomposition of initial correlation matrix in order to obtain 
        unique elements and flatten them for further purpose.

        Args:
            init_R (np.array, optional): Custom initital parameters. Defaults to None.
        """

        n = self.unif_data.shape[1]
        
        if init_R is None:
            init_R = np.corrcoef(self.unif_data, rowvar=False)
            
        chol_init_R = np.linalg.cholesky(init_R)
        init_params = chol_init_R.flatten()
        
        # set parameter bounds dynamically
        param_bounds = self.est_param.generate_param_bounds(len(init_params))

        opt = minimize(self.MLE_objective_fct, 
                       init_params, 
                       bounds=param_bounds,
                       constraints=self.est_param.cons)
        
        L = np.reshape(opt.x, (n, n))
    
        self.opt_param = L @ L.T
    

@dataclass
class CopulaTStudentHyperParam:
    
    # Two Stage Estimation Hyperparameters
    # Estimate correlation matrix first, degree of freedom later, 
    # analogously to gaussian copula
    
    cons_R_diag = [{'type': 'eq', 'fun': constraint_unit_diagonal},
                   {'type': 'eq', 'fun': constraint_is_positive_definite}]
    cons_nu_positive = [{'type': 'ineq', 'fun': lambda param: param}]
    
    def generate_param_bounds(n_coefs):
        """
        Correlation bounds
        """
        return [(-1,1)] * n_coefs
    
    
class CopulaTStudent():
    
    def __init__(self,
                 unif_data: Union[np.ndarray, pd.DataFrame]):
        self.unif_data = unif_data
        self.est_param = CopulaTStudentHyperParam
        self.opt_param = None
        
        
    @staticmethod
    def simulate_t_copula(shape: np.ndarray,
                          df: int,
                          sample_size: int, 
                          loc: np.array = None) -> np.array:
        
        n_asset = shape.shape[0]
        
        if loc is None:
            loc = np.zeros(n_asset)
        
        sim_cop_dependence = multivariate_t(loc = loc, 
                                            shape = shape, 
                                            df = df).rvs(size=sample_size)
        
        sim_cop_unif = np.column_stack([t.cdf(sim_cop_dependence[:,i], df = df)
                                        for i in range(n_asset)])
        
        return sim_cop_unif
        
    
    
    def log_density(self, 
                    R: np.ndarray, 
                    nu: int) -> float:
        """
        Compute the log-density of the t-copula using the inverse of the univariate 
        t-distribution.

        \Sigma is the correlation in this case!
        
        Raw Density:
        \( f(u; \Sigma, nu)=\Gamma\left(\frac{\nu + d}{2}\right)|\Sigma|^{-\frac{1}{2}} 
        \left[\Gamma\left(\frac{\nu + 1}{2}\right)\right]^{-d} 
        \left[1 + \frac{1}{\nu} z^T \Sigma^{-1} z \right]^{-\frac{\nu + d}{2}} 
        \prod_{i=1}^{d} \left[1 + \frac{z_i^2}{\nu} \right]^{-\frac{\nu + 1}{2}} \)

        Log-density:
        \( \log(f(u; \Sigma, \nu))=\log\left(\Gamma\left(\frac{\nu + d}{2}\right)\right)
        - \frac{1}{2} \log(|\Sigma|) 
        - d \log\left(\Gamma\left(\frac{\nu + 1}{2}\right)\right) 
        - \frac{\nu + 1}{2} \sum_{i=1}^{d} \log\left(1 + \frac{z_i^2}{\nu}\right)
        - \frac{\nu + d}{2} \log\left(1 + \frac{1}{\nu} z^T \Sigma^{-1} z\right) \)


        Args:
            u (Union[np.ndarray, pd.DataFrame]): A (n x d) matrix (numpy ndarray or 
                                                 pandas DataFrame) of copula data, where
                                                 n is the number of samples and d is 
                                                 the dimension.
            R (np.ndarray): A (d x d) numpy ndarray covariance matrix.
            nu (int): An int specifying the degrees of freedom.

        Returns:
            float: array of log-density evaluated at u.
        """


        if isinstance(self.unif_data, pd.DataFrame):
            self.unif_data = self.unif_data.values

        d = self.unif_data.shape[1]
        
        # this should be logged or and least throw a warning if values are clipped
        robust_data = np.clip(self.unif_data, a_min=1e-10, a_max=1-1e-10)
        
        # Transform u using the inverse CDF of the t-distribution
        z = t.ppf(robust_data, df=nu)

        L = cholesky(R, lower=True)
        det_Sigma = np.prod(np.diag(L))**2

        term1 = gammaln((nu + d) / 2)
        term2 = -0.5 * np.log(det_Sigma)
        term3 = -d * gammaln((nu + 1) / 2)

        inv_Lz = solve_triangular(L, z.T, lower=True)
        z_Sigma_inv_z = np.sum(inv_Lz**2, axis=0)

        term4 = -((nu + 1) / 2) * np.sum(np.log(1 + (z**2) / nu), axis=1)
        term5 = -(nu + d) / 2 * np.log(1 + (1 / nu) * z_Sigma_inv_z)

        # Sum the terms to get the log-density:
        log_density = term1 + term2 + term3 + term4 + term5
        
        return log_density


    def neg_copula_llh(self, R, nu) -> float:
            return -np.sum(self.log_density(R = R, nu = nu))
        
    
    def MLE_objective_fct_R(self, param: np.array, nu_static) -> float:
        """
        MLE objective function for correlation parameters only
        
        """
        
        n = self.unif_data.shape[1] 
        L = param.reshape(n, n)
        R = L @ L.T
    
        return self.neg_copula_llh(R, nu_static)
        
        
    def MLE_objective_fct_nu(self, param: np.array, R_static) -> float:
        """
        MLE objective function for degree of freedom estimation only

        """
        return self.neg_copula_llh(R_static, param)
    
    
    def _fit_R(self, init_R = None, static_nu = None):
        """
        Optimize data w.r.t. correlation coefficient only. Used in a two stage
        estimation approach
        """
        
        d = self.unif_data.shape[1]
        
        if init_R is None:
            init_R = np.corrcoef(self.unif_data, rowvar=False)
            
        init_R_chol = np.linalg.cholesky(init_R)
        init_R_params = init_R_chol.flatten()
        
        if static_nu is None:
            static_nu = d
        
        param_bounds = self.est_param.generate_param_bounds(len(init_R_params))
        
        opt_R = minimize(self.MLE_objective_fct_R, 
                         init_R_params, 
                         bounds=param_bounds,
                         args = static_nu,
                         constraints = self.est_param.cons_R_diag)
    
        opt_R = opt_R.x
        opt_R = np.reshape(opt_R, (d, d))
        opt_R = opt_R @ opt_R.T

        return opt_R
    
    
    def _fit_nu(self, init_nu = None, static_R = None):
        """
        Optimize data w.r.t. degree of freedom. Static correlation matrix
        is the optimized correlation matrix from _fit_R method or in simulation
        context maybe obtained from a dcc model.
        """
        
        n = self.unif_data.shape[0]
        
        if init_nu is None:
            init_nu = n
        
        opt_nu = minimize(self.MLE_objective_fct_nu, 
                          init_nu,
                          bounds = [(1, np.inf)],
                          args = static_R,
                          constraints = self.est_param.cons_nu_positive)   
        
        opt_nu = opt_nu.x
        
        return opt_nu
    
    
    def fit(self, init_R = None, init_nu = None):
        """
        Estimate parameters of the t-copula in two steps. Correlation
        matrix first, tail heavyness last. See description in one stage
        estimation.
        
        """
        
        opt_R = self._fit_R(init_R = init_R)
        opt_nu = self._fit_nu(init_nu = init_nu, static_R = opt_R)
        
        self.opt_param = (opt_R, opt_nu)
    