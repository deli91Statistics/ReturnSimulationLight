import unittest
import numpy as np
import pandas as pd

from scipy.stats import norm

from src2.variance_models import GARCH, GJRGARCH


"""
Test Catalogue
1 correct length of array
2 no NaN values
3 non-negative values
4 stationary requirement

"""


class TestGARCH(unittest.TestCase):
    
    def setUp(self):
        self.data = pd.Series(np.random.randn(100))
        self.variance_model = GARCH(self.data)
        self.params = [0.1, 0.2, 0.7]
        
        
    def test_recursion(self):

        sigma_2 = self.variance_model.variance_recursion(data = self.data,
                                                      params = self.params)
        
        self.assertEqual(len(sigma_2), len(self.data))  # 1
        self.assertFalse(np.any(np.isnan(sigma_2)))     # 2
        self.assertTrue(all(s > 0 for s in sigma_2))    # 3

        
       
    def test_create_param_grid(self):
        
        grid = self.variance_model.create_param_grid()
        
        self.assertTrue(all(all(item > 0 for item in row) for row in grid))     # 1
        self.assertTrue(all((row[-2] + row[-1]) < 1 for row in grid ))          # 4
        
        
    def test_simulation(self):
        
        eps = norm.rvs(size=300)
        
        resid, sigma_2 = self.variance_model.simulate(params=self.params,
                                                      innovations=eps)
        
        self.assertEqual(len(sigma_2), len(eps))        # 1
        self.assertEqual(len(resid), len(eps))          # 1
        self.assertFalse(np.any(np.isnan(sigma_2)))     # 2
        self.assertTrue(all(s > 0 for s in sigma_2))    # 3


    
class TestGJRGARCH(unittest.TestCase):
    
    def setUp(self):
        self.data = pd.Series(np.random.randn(100))
        self.variance_model = GJRGARCH(self.data)
        self.params = [0.1, 0.2, 0.1, 0.7]
        
        
    def test_recursion(self):
        
        sigma_2 = self.variance_model.variance_recursion(data = self.data,
                                                      params = self.params)
        
        self.assertEqual(len(sigma_2), len(self.data))  # 1
        self.assertFalse(np.any(np.isnan(sigma_2)))     # 2        
        self.assertTrue(all(s > 0 for s in sigma_2))    # 3

        
        
    def test_create_param_grid(self):
        
        grid = self.variance_model.create_param_grid()
        
        self.assertTrue(all(all(item > 0 for item in row) for row in grid))          # 3
        self.assertTrue(all((row[-3]+ 0.5*row[-2] + row[-1]) < 1 for row in grid ))  # 4
    
    
    def test_simulation(self):
        
        eps = norm.rvs(size=300)
        
        resid, sigma_2 = self.variance_model.simulate(params=self.params,
                                                      innovations=eps)
        
        self.assertEqual(len(sigma_2), len(eps))        # 1
        self.assertEqual(len(resid), len(eps))          # 1
        self.assertFalse(np.any(np.isnan(sigma_2)))     # 2
        self.assertTrue(all(s > 0 for s in sigma_2))    # 3
        