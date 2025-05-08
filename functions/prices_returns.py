import pandas as pd
import numpy as np
from typing import Union


def price_to_net_returns(price: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the percentage change of inserted price dataframe.
    Discard NaN entries.

    Args:
        data (Union[pd.DataFrame, pd.Series]): price data

    Returns:
        pd.DataFrame: raw (net) returns
    """
    
    return price.pct_change().dropna()


def price_to_log_returns(price: pd.DataFrame) -> pd.DataFrame:
    """
    Compute logarithmic returns of financial returns.
    Recall, log-returns are almost equal to raw returns if close to zero

    Args:
        data (pd.DataFrame): price data

    Returns:
        pd.DataFrame: log returns
    """
    
    return np.log(price).diff().dropna()


def aggregate_log_returns(log_returns: pd.DataFrame) -> float:
    """
    Aggregate logarithmic returns. Recall that log returns are additive

    Args:
        log_returns (pd.DataFrame): log returns
    """
    
    return np.sum(log_returns)


def aggregate_net_returns(net_returns: pd.DataFrame) -> float:
    """
    Aggregate net returns by converting it to gross returns first,
    multiplying all periods and subtract 1 at the end.
    
    Args:
        net_returns (pd.DataFrame): net returns

    Returns:
        float: aggregated multiple period net returns
    """
    
    gross_returns = 1 + net_returns
    agg_gross_returns = np.prod(gross_returns)
    agg_net_return = agg_gross_returns - 1
    
    return agg_net_return


def calculate_returns(price: pd.DataFrame, 
                      frequency: str, 
                      log_returns: bool) -> pd.DataFrame:
    """    
    Calculates simple or logarithmic daily, weekly, monthly, yearly returns based on a 
    Pandas DataFrame containing price data.
    
    Args:
        price (pd.DataFrame): Price data
        frequency (Frequency): The frequency of the returns calculation
        log_returns (bool, optional): Whether to calculate logarithmic returns

    Raises:
        ValueError: Only D, W, M, Q, Y allowed as frequency

    Returns:
        pd.DataFrame: _description_
    """

    if log_returns:
        returns = price_to_log_returns(price = price)
        method = aggregate_log_returns
    else:
        returns = price_to_net_returns(price = price)
        method = aggregate_net_returns


    # Calculate the returns based on the specified frequency
    if frequency == 'D':
        returns = returns
    elif frequency == 'W':
        returns = returns.resample('W').apply(method)
    elif frequency == 'M':
        returns = returns.resample('M').apply(method)
    elif frequency == 'Q':
        returns = returns.resample('Q').apply(method)
    elif frequency == 'Y':
        returns = returns.resample('Y').apply(method)
    else:
        raise ValueError("Invalid frequency. Choose only D, W, M, Q, Y")

    return returns


def return_to_prices(returns: Union[pd.DataFrame, pd.Series, np.array], 
                     init_price: Union[float, list, np.array],
                     log_returns: bool) -> Union[pd.DataFrame, np.ndarray]:
    """
    Converts returns to prices.
    
    Recall that log returns are defined as
        
        ln(1 + R_t) = ln(P_t / P_{t-1})
    
    where R_t is the net return. Therefore, we need to apply the exponential function
    on the log returns first.
    
    Given net returns, convert them to gross returns by adding one to all entries

    Args:
        log_returns (Union[pd.DataFrame, np.ndarray]): logarithmic returns
        initial_price (float): initial price, usually closing price of the previous day

    Returns:
        Union[pd.DataFrame, np.ndarray]: price data
    """
    
    T = returns.shape[0]
    
    if isinstance(returns, pd.DataFrame):
        returns = returns.values
        n = returns.shape[1]
        prices = np.zeros((T+1, n))
    elif isinstance(returns, pd.Series):
        returns = returns.values
        prices = np.zeros((T+1, ))
    elif isinstance(returns, np.array):
        prices = np.zeros((T+1, ))
    else:
        raise ValueError("Return must be of type, pd.DataFrame, pd.Series or np.array")
    
        
    if log_returns:
        r = np.exp(returns)
    else:
        r = returns + 1
    
    
    prices[0] = init_price
    

    for t in range(1, T + 1):
        prices[t] = prices[t - 1] * r[t - 1]
    
    return prices
    