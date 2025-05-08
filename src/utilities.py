import pandas as pd
from scipy.stats import ecdf


from functions.distributions import ErrorDistribution, Gaussian, TStudent
from src.stochastic_volatility import VarianceProcess, GARCH, GJRGARCH


# This module serves as a helper module for the cockbook (for now)
# the majority of the models are univariate and targeted to treat one time series only
# looping over the assets requires to save the results. This model helps to process
# the results.




# Handling univariate variance models
    
def estimate_variance_models(data: pd.DataFrame, 
                             VarianceProcess: VarianceProcess,
                             ErrorDistribution: ErrorDistribution):

    univariate_volatility_models = {}
    
    for ticker, data in data.items():
        VarianceModel = VarianceProcess(data = data, 
                                        ErrorDistribution = ErrorDistribution)
        VarianceModel.fit()
        univariate_volatility_models[ticker] = VarianceModel
    
    return univariate_volatility_models


def get_optimal_parameters(univariate_volatility_models: dict):
    
    param_lst = [univariate_volatility_models[key].opt_param 
                 for key in univariate_volatility_models.keys()]
    
    df = pd.DataFrame(param_lst, index=univariate_volatility_models.keys())
    return df


def get_predicted_variance(univariate_volatility_models: dict):
    
    model_dict = {ticker: var_mod.pred_variance 
                  for ticker, var_mod in univariate_volatility_models.items()}
    
    return pd.DataFrame(model_dict)


def get_standardized_residuals(univariate_volatility_models: dict):
    
    std_resid_dict = {ticker: var_mod.std_residuals 
                      for ticker, var_mod in univariate_volatility_models.items()}
    
    return pd.DataFrame(std_resid_dict)



# Transforming for copulas

def create_sample_ecdf(data: pd.DataFrame):

    ecdf_model = {ticker: ecdf(data) for ticker, data in data.items()}
    
    return ecdf_model


def convert_to_uniform_ecdf(data: pd.DataFrame):
    
    ecdf_models_dict = create_sample_ecdf(data)

    unif_data_dict = {}
    
    for ticker, data in data.items():
        unif_data = ecdf_models_dict[ticker].cdf.evaluate(data)
        unif_data_dict[ticker] = unif_data
    
    return pd.DataFrame(unif_data_dict)
