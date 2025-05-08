import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass

from scipy.optimize import minimize
from typing import Optional, Union, Protocol

from functions.distributions import ErrorDistribution



class VarianceProcess(Protocol):
    """
    Protocol for VarianceProcess typing, used for example for FHS
    
    """
    
    opt_param = ...
    pred_variance = ...
    std_residuals = ...
    
    def fit(self) -> None: ...
    def compute_predicted_variance(self) -> None: ...
    def compute_standardized_residuals(self) -> None: ...



@dataclass
class GARCHEstHyperParam():
    # Hyperparameters for grid search
    arch_grid_bd = [0.001, 0.4]
    garch_grid_bd = [0.5, 0.98]
    grid_resolution = 30
    
    # Hypterparameters for minimize routine
    param_bounds = [(0,1), (0,1), (0,1)]
    param_constraints = [{'type': 'ineq', 'fun': lambda param: -(param[1]+param[2]-1)},
                         {'type': 'ineq', 'fun': lambda param: param[1]},
                         {'type': 'ineq', 'fun': lambda param: param[2]}]


class GARCH(VarianceProcess):
    """
    This class optimizes a gaussian likelihood function given conditional
    variances (sigma2) which is specified by a GARCH(1,1) model.
    
    The estimation is done in the following way:
    
    1) A grid of the three parameters (const, arch, garch) or (omega, alpha, beta)
       are generated.
    2) The negative likelihood function is evaluated based on the data and the
       conditional variances for all triplets of parameters from the grid. 
       The parameter set which yields the minimum likelihood value are stored
    3) Perform an optimization routine with initial values obtained from 2)
    
    At the moment, only gaussian innovations are possible
    
    """

    def __init__(self,
                 data: Union[pd.Series, np.array], 
                 ErrorDistribution: ErrorDistribution):
        
        assert data.ndim == 1, 'Only univariate data allowed'
        
        self.data = data            # Univariate Data!!!
        self.name = 'GARCH'
        self.ErrorDistribution = ErrorDistribution
        self.est_param = GARCHEstHyperParam()
        self.opt_param = None
        self.pred_variance = None
        self.std_residuals = None
        

    @staticmethod
    def garch_11_recursion(data: pd.Series,
                           const: float, 
                           arch_coef: float,
                           garch_coef: float, 
                           init_sigma2: Optional[None] = None) -> np.array:
        """
        Compute conditional variance of GARCH(1,1) model recursively. Recall the GARCH
        model recursion
        
        sigma_t^2 = omega + alpha_1 r_{t-i}^2 + beta_1 sigma_{t-j}^2

        Args:
            const (float): omega
            arch_coef (float): alpha_1
            garch_coef (float): beta_1
            resid (Union[np.array, pd.Series]): innovations
            init_sigma2 (Optional[None], optional): starting variance. Defaults to None.

        Returns:
            np.array: conditional volatility series
        """

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
        
    
    def neg_likelihood_fct(self, params: list) -> float:
        """
        Computes the value of the likelihood function with conditional variance. The
        variance recursion has to be included otherwise the optimization does not
        work. 
        
        Given a normal distribution, this likelihood can be extended by conditional
        means in the same ways as it was done for the variance. Simply implement
        a mean recursion, e.g. AR, MA, ARMA, etc..

        Args:
            params (list): list of garch parameters, 
            i.e. const. (omega), arch parameter (alhpa_1), garch parameter (beta_1) 

        Returns:
            float: Evaluated likelihood function
        """

        # Note that this is a zero mean model
        
        
        # adjust parameters if error distributions also have parameters by keywords
        if self.ErrorDistribution.param_count > 0:
            dist_param = {p_name: p_val 
                          for p_name, p_val in zip(self.ErrorDistribution.param_names,
                                                   params[-self.ErrorDistribution.param_count:])}
        else:
            dist_param = {}
        
        
        sigma2 = self.garch_11_recursion(data = self.data,
                                         const = params[0],
                                         arch_coef = params[1],
                                         garch_coef = params[2])
        
        
        lls = self.ErrorDistribution.log_density(self.data, 
                                                 sigma2=sigma2,
                                                 **dist_param)
        
        return -np.sum(lls)
    
    
    def _create_garch_param_grid(self) -> list:
        """
        Generates a list of parameters as a pre-selection of starting
        values for the subsequent likelihood optimization. There is
        no clear theory on how to estimate a likelihood function associated
        with a garch model efficiently. Thus, the grid is generated based on
        pure experience. See ARCH package of Kevin Sheppard.

        Args:
            data_series (Union[pd.Series, np.array]): Data Series
            arch_grid_bounds (list): upper and lower bound of arch component,
                                        sometimes alpha is used as a symbol
            garch_grid_bounds (list): upper and lower bound of garch component,
                                        sometimes beta is used as a symbol
            resolution (int): Controls how fine the grid is. It sets the spacing
                                between the lower and upper bound

        Returns:
            list: Returns a list of starting values. Each entry consists of three
                    possible parameters
        """

        assert len(self.est_param.arch_grid_bd) == 2,  "Set bound, e.g. [0.001, 0.5]"
        assert len(self.est_param.garch_grid_bd) == 2, "Set bound, e.g. [0.5, 0.999]"

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
            
            # Adj number of parameters by the parameters of the distribution
            # 5 was randomly choosen, as starting value for df if tstudent is selected
            # or for both skewness or df given skewed student-t
            sv = np.ones(3 + self.ErrorDistribution.param_count) * 5
            pre_arch, pre_garch = values
            sv[0] = const * (1 - pre_garch)     # omega
            sv[1] = pre_arch                    # alpha_1
            sv[2] = pre_garch - pre_arch        # beta_1
            
            lst_starting_parameters.append(sv)

        return lst_starting_parameters
    
    
    def _search_param_grid(self) -> list:
        """
        Computes minimal value of likelihood function given a grid of parameters.
        This is used as a pre-selection of initial values for optimizing the 
        actual likelihood function.

        Note that the NEGATIVE likelihood function is evaluated.
        
        Returns:
            list: List of "optimal" parameters as starting values
        """
        
        param_grid = self._create_garch_param_grid()
        
        neg_llhs = []

        for param in param_grid:
            neg_llh = self.neg_likelihood_fct(params=param)
            neg_llhs.append(neg_llh)
            
        return param_grid[np.argmin(neg_llhs)]


    def fit(self):
        """
        Fit the model and save optimal parameters, predicted variance
        and standardized residuals.
        """

        init_param = self._search_param_grid()
        
        # adjust bounds if distribution has additional parameter bounds
        if self.ErrorDistribution.param_count > 0:
            joint_bounds = self.est_param.param_bounds + self.ErrorDistribution.param_bounds
        else:
            joint_bounds = self.est_param.param_bounds
        
        
        print(init_param)
        
        opt = minimize(self.neg_likelihood_fct, 
                       init_param, 
                       bounds=joint_bounds,
                       constraints=self.est_param.param_constraints,
                       method="SLSQP", 
                       tol= None,
                       options=None)
            
        self.opt_param = opt.x
        
        self.predicted_variance()
        self.standardized_residuals()    
                    
    
    def predicted_variance(self) -> np.array:
        """
        Returns the (predicted) variance based on optimized conditional
        variance parameters.

        Returns:
            np.array: Predicted variance
        """
        
        predicted_variance = self.garch_11_recursion(data = self.data,
                                                     const = self.opt_param[0],
                                                     arch_coef = self.opt_param[1],
                                                     garch_coef = self.opt_param[2])
        
        self.pred_variance = pd.Series(index=self.data.index, data = predicted_variance)
        
        return self.pred_variance
    
    
    def standardized_residuals(self) -> np.array:
        """
        Computes standardized residuals. Recall
        
        Given a GARCH(1,1) model the standardized residuals
        are obtained by deviding the returns by the volatility.
        
        $ \varepsilon = r_t / sigma_t $
        
        If the models captures heteroscedasticity accordlingy,
        the residuals should be white noise.

        Returns:
            np.array: Array of standardized residuals
        """
        
        sigma2 = self.predicted_variance()
        sigma = np.sqrt(sigma2)
        
        self.std_residuals = self.data/sigma
        
        return self.std_residuals
        
        
    def forecast(self, steps: int = 1) -> np.array:
        """
        Forecast the conditional variance for the given number of steps ahead.

        Args:
            steps (int): Number of time periods to forecast.

        Returns:
            np.array: Array of forecasted variances.
        """

        # Initial values based on the last observed data and variance
        last_data = self.data[-1]
        last_variance = self.pred_variance.iloc[-1]

        forecasted_variances = []

        for _ in range(steps):
            # Use the GARCH(1,1) recursion to forecast the next variance
            next_variance = (self.opt_param[0] + 
                            self.opt_param[1] * last_data**2 + 
                            self.opt_param[2] * last_variance)

            # Append the forecasted variance to the list
            forecasted_variances.append(next_variance)

            # Update the "last" values for the next iteration
            # As we don't know the future returns, assume it to be zero for simplicity
            last_data = 0  
            last_variance = next_variance

        return np.array(forecasted_variances)
    
    
    def simulate(self, n_sim: int = None,
                 const: float = None, arch_coef: float = None, garch_coef: float = None, 
                 sigma2_0: float = None, innovations: np.array = None):
        """
        Simulate a GARCH model with specific inovations. If no innovation process is
        provided, sample from a standard normal distribution.

        Args:
            n_sim (int, optional): number of simulation size, if no innovation process 
                                   is provided. Defaults to None.
            const (float, optional): omega. Defaults to None.
            arch_coef (float, optional): alpha1. Defaults to None.
            garch_coef (float, optional): beta1. Defaults to None.
            sigma2_0 (float, optional): initial variance. Defaults to None.
            innovations (np.array, optional): innovarion process. Defaults to None.

        Returns:
            tupel(np,array, np.array): tuple of numpy arrays containing simualted variance
                                       and simulated returns
        """

        if const is None:
            const = self.opt_param[0]
        if arch_coef is None:
            arch_coef = self.opt_param[1]
        if garch_coef is None:
            garch_coef = self.opt_param[2]


        if innovations is None:
            eps = np.random.normal(0, 1, n_sim)
        else:                       
            eps = innovations
            n_sim = len(eps)
            
                    
        r = np.zeros(n_sim)
        sigma2 = np.zeros(n_sim)


        if sigma2_0 is None:
            sigma2[0] = const / (1 - arch_coef - garch_coef)
        else:
            sigma2[0] = sigma2_0
        
        for i in range(1, n_sim):
            sigma2[i] = const + arch_coef * (r[i - 1] ** 2) + garch_coef * sigma2[i - 1]
            r[i] = np.sqrt(sigma2[i]) * eps[i]
            
        return r, sigma2
        

@dataclass
class GJRGARCHEstHyperParam():
    # Hyperparameters for grid search
    arch_grid_bd = [0.001, 0.4]
    garch_grid_bd = [0.5, 0.98]
    grid_resolution = 30
    
    # Hypterparameters for minimize routine
    param_bounds = [(0,1), (0,1), (0,1), (0,1)]
    param_constraints = [
        {'type': 'ineq', 'fun': lambda param: -(param[1]+0.5*param[2]+param[3]-1)},
        {'type': 'ineq', 'fun': lambda param: param[1]},
        {'type': 'ineq', 'fun': lambda param: param[2]},
        {'type': 'ineq', 'fun': lambda param: param[3]}
        ]


class GJRGARCH(VarianceProcess):
    def __init__(self,
                 data: Union[pd.Series, np.array], 
                 ErrorDistribution: ErrorDistribution):
        
        assert data.ndim == 1, 'Only univariate data allowed'
    
        self.data = data            # Univariate Data!!!
        self.ErrorDistribution = ErrorDistribution
        self.est_param = GJRGARCHEstHyperParam()
        self.opt_param = None
        self.pred_variance = None
        self.std_residuals = None
        
    
    @staticmethod
    def gjr_garch_11_recursion(data: pd.Series,
                               const: float, 
                               arch_coef: float,
                               lev_coef: float,
                               garch_coef: float, 
                               init_sigma2: Optional[None] = None) -> np.array:
        
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
    
    
    def neg_likelihood_fct(self, params: list) -> float:
        
        
        if self.ErrorDistribution.param_count > 0:
            dist_param = {p_name: p_val 
                          for p_name, p_val in zip(self.ErrorDistribution.param_names,
                                                   params[-self.ErrorDistribution.param_count:])}
            
        else:
            dist_param = {}
            
        
        sigma2 = self.gjr_garch_11_recursion(data = self.data,
                                             const = params[0],
                                             arch_coef = params[1],
                                             lev_coef = params[2],
                                             garch_coef = params[3])
        
        lls = self.ErrorDistribution.log_density(self.data, sigma2=sigma2, **dist_param)
        
        return -np.sum(lls)
    
    
    
    def _create_gjr_garch_param_grid(self):
        
        # ARCH, GARCH, Leverage component grid
        alpha = np.linspace(self.est_param.arch_grid_bd[0], 
                            self.est_param.arch_grid_bd[1], 
                            num=self.est_param.grid_resolution)
                            
        beta = np.linspace(self.est_param.garch_grid_bd[0],
                           self.est_param.garch_grid_bd[1], 
                           num=self.est_param.grid_resolution)
    
        gamma = alpha
        
        grid = list(itertools.product(*[alpha, beta, gamma]))

        # Starting value for baseline volatility
        const = np.mean(abs(self.data)**2)
        
        # Construct grid with constraints
        lst_starting_parameters = []
        for values in grid:
            sv = np.ones(4 + self.ErrorDistribution.param_count) * 5
            pre_alpha, pre_beta, pre_gamma  = values
            sv[0] = const * (1 - pre_beta)                  # omega
            sv[1] = pre_alpha                               # alpha_1
            sv[2] = pre_gamma                               # gamma_1
            sv[3] = pre_beta - pre_alpha - 0.5*pre_gamma    # beta_1
            
            lst_starting_parameters.append(sv)

        return lst_starting_parameters
    
    
    
    def _search_param_grid(self) -> list:
         
        param_grid = self._create_gjr_garch_param_grid()
        
        neg_llhs = []

        for param in param_grid:
            neg_llh = self.neg_likelihood_fct(params=param)
            neg_llhs.append(neg_llh)
            
        return param_grid[np.argmin(neg_llhs)]
    
    
    
    def fit(self):

        init_param = self._search_param_grid()
        
        # adjust bounds if distribution has additional parameter bounds
        if self.ErrorDistribution.param_count > 0:
            joint_bounds = self.est_param.param_bounds + self.ErrorDistribution.param_bounds
        else:
            joint_bounds = self.est_param.param_bounds
        
        
        
        opt = minimize(self.neg_likelihood_fct, 
                       init_param, 
                       bounds=joint_bounds,
                       constraints=self.est_param.param_constraints,
                       method="SLSQP", 
                       tol= None,
                       options=None)
            
        self.opt_param = opt.x
        
        self.predicted_variance()
        self.standardized_residuals()    
    
    
    def predicted_variance(self) -> np.array:
        """
        Returns the (predicted) variance based on optimized conditional
        variance parameters.

        Returns:
            np.array: Predicted variance
        """
        
        predicted_variance = self.gjr_garch_11_recursion(data = self.data,
                                                         const = self.opt_param[0],
                                                         arch_coef = self.opt_param[1],
                                                         lev_coef = self.opt_param[2],
                                                         garch_coef = self.opt_param[3])
        
        self.pred_variance = pd.Series(index=self.data.index, data = predicted_variance)
        
        return self.pred_variance
    
    
    def standardized_residuals(self) -> np.array:
        """
        Computes standardized residuals. Recall
        
        Given a GARCH(1,1) model the standardized residuals
        are obtained by deviding the returns by the volatility.
        
        $ \varepsilon = r_t / sigma_t $
        
        If the models captures heteroscedasticity accordlingy,
        the residuals should be white noise.

        Returns:
            np.array: Array of standardized residuals
        """
        
        sigma2 = self.predicted_variance()
        sigma = np.sqrt(sigma2)
        
        self.std_residuals = self.data/sigma
        
        return self.std_residuals