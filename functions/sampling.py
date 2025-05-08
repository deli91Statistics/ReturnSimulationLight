import pandas as pd
import numpy as np
from typing import Union

def create_bootstrap_sample(sample: Union[pd.DataFrame, pd.Series, np.ndarray], 
                            sample_size: int) -> Union[pd.DataFrame, np.ndarray]:
    """
    Create bootstrap sample of specified size by creating a random array of row indices
    first and then select from the population sample. Each iterator thus draws an entire
    row if a multidimensional dataset is given.

    Args:
        sample (Union[pd.DataFrame, pd.Series, np.ndarray]): Sample to sample from.
        sample_size (int): size of the datasample. Usually the same size as
                           the data from which is sampled from.

    Returns:
        Union[pd.DataFrame, np.ndarray]: Sampled data.
    """
    

    if isinstance(sample, pd.DataFrame):
        sample_array = sample.values
    elif isinstance(sample, pd.Series):
        sample_array = sample.values
    else:
        sample_array = sample

    bootstrap_index = np.random.randint(0, len(sample_array), size=sample_size)

    # Check if array is one-dimensional
    if sample_array.ndim == 1:
        bootstrap_sample = sample_array[bootstrap_index]
    else:
        bootstrap_sample = sample_array[bootstrap_index, :]
    
    return bootstrap_sample
