import numpy as np
from numpy.random import RandomState

from typing import Union, Optional
from abc import ABCMeta, abstractmethod

from scipy.special import gammaln


# This is also part of a personal exercise using different abstraction
# methods. Here, the template design pattern is used. An abstract
# base class (ABC) is defined and all distribution objects inherit from this
# base class. Check the stochastic volatility module where a Protocol is used
# instead



class ErrorDistribution(object, metaclass=ABCMeta):
    """
    Template for subclassing

    Instance variables:

    - _name : Name of the innovation distribution
    - random_state : RandomState object for reproducibility
    
    If you wish to set a specific random state, import the random module
    from numpy first and pass a random state object when instancing a 
    innovation distribuion object:
    
    '''    
        from numpy.random import RandomState
        
        A = Gaussian(RandomState(42))
    '''
    
    Assigning Gaussian(RandomState(42)) to another variable produces the same
    results when generating random numbers from the respective distribution.
    

    Public methods:

    - log_density() :  Logarithmic density function
    - generate_std_innovations() :  generate std. random variates depending on
                                    depending on the distribution selected
                                    e.g. normal = standard normal, etc.
    
    """
    
    def __init__(self,
                 random_state: Optional[RandomState] = None,
    ) -> None:
        self._name = 'Distribution'
        self.random_state = random_state
        self.param_count = None
        self.param_name = None
        self.param_bounds = None
        
    
    @staticmethod
    def log_density(data: Union[float, np.array], 
                    sigma2: Union[float, np.array],
                    ) -> Union[float, np.array]:
        """
        Compute the probability mass given a datapoint at a single location.

        Args:
            data (Union[float, np.array]): data series, e.g. financial returns
            sigma2 (Union[float, np.array]): variance series, e.g. from
                                             volatily models (GARCH, etc.)

        Returns:
            Union[float, np.array]: value or sequence of prob. mass
        """
        pass
    
    @abstractmethod
    def generate_std_innovations(self, size: int) -> Union[float, np.array]:
        """
        Generate standard shocks given a specific sample size

        Args:
            size (int): size of the sample

        Returns:
            Union[float, np.array]: array of standard innovation variates
        """
        pass


    @staticmethod
    def multivariate_log_density(data: np.array,
                                 cov_mat: np.array) -> np.array :
        pass
    


class Gaussian(ErrorDistribution):
    
    def __init__(self, 
                 random_state: Optional[RandomState] = None):
        
        super().__init__(random_state=random_state)
        self._name = 'Gaussian'
        self.param_count = 0


    @staticmethod
    def log_density(data: Union[float, np.array], 
                    sigma2: Union[float, np.array]) -> Union[float, np.array]:
        
        # this log_density is only suitable for estimating conditional variance
        # tbd: add variable mu for conditional mean
        
        if isinstance(sigma2, float):
            assert sigma2 > 0
        elif isinstance(sigma2, np.ndarray):
            assert (sigma2 > 0).any()
        
        return -0.5*(np.log(2*np.pi) + np.log(sigma2) + (data**2.0)/sigma2)


    def generate_std_innovations(self, size: int) -> Union[float, np.array]:
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        
        gaussian_shocks = self.random_state.normal(size = size)
        return gaussian_shocks
    
    
    @staticmethod
    def multivariate_log_density(data: np.array, cov_mat: np.array) -> float:
        """
        This is the robust implementation of a multivariate log-normal density.
        This is necessary for avoiding computation errors, numerical instabilities
        and efficiency. The code looked like this:
        
        
        def multivariate_log_density(data: np.array,
                                     cov_mat: np.array) -> float :
    
        inv_cov = np.linalg.inv(cov_mat) 
        eval_density = -0.5*(np.log(np.linalg.det(cov_mat)) + data @ inv_cov @ data.T)
        
        return eval_density
    

        Args:
            data (np.array): data series
            cov_mat (np.array): covariance matrix

        Returns:
            float: Probability mass of the data vector
        """
        # Decompose the covariance matrix using Cholesky decomposition
        L = np.linalg.cholesky(cov_mat)
        
        # Avoid explicit inversion: solve for inv_cov @ data
        sol = np.linalg.solve(L, data)
        
        # Compute the squared magnitude of sol (which equals to data @ inv_cov @ data.T)
        quadratic_form = np.sum(sol**2)
        
        # Calculate the log determinant using the Cholesky factor
        log_det = 2 * np.sum(np.log(np.diag(L)))
    
        return -0.5 * (log_det + quadratic_form)

    
    
class TStudent(ErrorDistribution):
    
    def __init__(self, 
                 df: int = None,
                 random_state: Optional[RandomState] = None):
    
        super().__init__(random_state=random_state)
        self._name = 'T-Student'
        self.df = df
        self.param_count = 1
        self.param_names = ['df']
        self.param_bounds = [(1, np.inf)]
        
        
    @staticmethod
    def log_density(data: Union[float, np.array], 
                    sigma2: Union[float, np.array],
                    df: int) -> Union[float, np.array]:
        
        lls = gammaln((df+1)/2)-gammaln(df/2)-np.log(np.pi*(df-2))/2
        lls -= 0.5*(np.log(sigma2))
        lls -= ((df+1)/2) * (np.log(1 + (data**2.0)/(sigma2*(df-2))))
        
        return lls


    def generate_std_innovations(self, size: int) -> Union[float, np.array]:
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        
        gaussian_shocks = self.random_state.standard_t(df=self.df, size=size)
        return gaussian_shocks
    
   