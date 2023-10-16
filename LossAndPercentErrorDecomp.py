import logging
import numpy as np
import torch
from Visualizer.SeriesDecomp import series_decomp  
from utility import compute_percent_error

def compute_mse_and_errors(pred, y, period, seasonal, bias):
    # Decompose data
    trend_y, seasonal_y, resid_y = series_decomp(y.cpu().numpy(), period, seasonal)
    trend_pred, seasonal_pred, resid_pred = series_decomp(pred.cpu().detach().numpy(), period, seasonal)

    # Convert np.array to tensor - gradient will be removed. Thus, MSE and percent_error for trend, seasonal, resid are not backpropagatable.
    criterion = torch.nn.MSELoss()
   
    original_loss = criterion(pred, y)
    trend_loss = criterion(torch.tensor(trend_pred), torch.tensor(trend_y))
    seasonal_loss = criterion(torch.tensor(seasonal_pred), torch.tensor(seasonal_y))
    resid_loss = criterion(torch.tensor(resid_pred), torch.tensor(resid_y))
    
    percent_error_original = compute_percent_error(pred, y, bias) 
    percent_error_trend = compute_percent_error(torch.tensor(trend_pred), torch.tensor(trend_y), bias) 
    percent_error_seasonal = compute_percent_error(torch.tensor(seasonal_pred), torch.tensor(seasonal_y), bias)
    percent_error_resid = compute_percent_error(torch.tensor(resid_pred), torch.tensor(resid_y), bias) 

    logging.info(f'original_loss: {original_loss.item():.4f} [(kW/hour)^2]')   
    logging.info(f'trend_loss: {trend_loss.item():.4f} [(kW/hour)^2]')
    logging.info(f'seasonal_loss: {seasonal_loss.item():.4f} [(kW/hour)^2]')
    logging.info(f'resid_loss: {resid_loss.item():.4f} [(kW/hour)^2]\n')

    logging.info(f'percent_error_orignal: {percent_error_original.item():.4f} [%/hour]')   
    logging.info(f'percent_error_trend: {percent_error_trend.item():.4f} [%/hour]')
    logging.info(f'percent_error_seasonal: {percent_error_seasonal.item():.4f} [%/hour]')
    logging.info(f'percent_error_resid: {percent_error_resid.item():.4f} [%/hour]\n')

