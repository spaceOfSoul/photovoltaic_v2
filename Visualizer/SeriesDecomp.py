from statsmodels.tsa.seasonal import STL
import torch
import numpy as np

def series_decomp(data, period, seasonal):
    """Decompose data into trend, seasonal, residual components"""

    stl = STL(data, period=period, seasonal=seasonal)
    stl = stl.fit()
    
    trend = stl.trend
    seasonal = stl.seasonal
    residual = stl.resid

    return trend, seasonal, residual

