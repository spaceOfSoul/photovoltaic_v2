import os
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import logging
from Visualizer.SeriesDecomp import series_decomp  

def path2date(path):
    date_str = path.split('/')[-2] + '/' + \
        path.split('/')[-1].replace('.xlsx', '')
    date = datetime.strptime(date_str, '%Y_%m/%d')
    return date


def list_up_solar(solar_directory):
    # print("Current working directory in list_up_solar:", os.getcwd())
    # build photovoltaic data list
    solar_dir = os.listdir(solar_directory)
    # solar_dir.sort()
    solar_list = []
    for folder in solar_dir:
        mlist = os.listdir(solar_directory+'/'+folder)
        mlist = [file for file in mlist if file.find('xlsx') > 0]
        mlist = sorted(mlist, key=lambda x: int(x.split('.')[0]))
        for file in mlist:
            path = solar_directory + '/' + folder + '/' + file
            solar_list.append(path)

    solar_list.sort(key=path2date)

    # find period
    first_ = solar_list[0].split('.')[1].split('/')
    first_year, first_month = first_[-2].split('_')
    first_day = str("%02d" % int(first_[-1]))
    first_month = str("%02d" % int(first_month))###
    first_date = first_year+first_month+first_day

    last_ = solar_list[-1].split('.')[1].split('/')
    last_year, last_month = last_[-2].split('_')
    last_day = str("%02d" % int(last_[-1]))
    last_month = str("%02d" % int(last_month))###
    last_date = last_year+last_month+last_day
    #print('Training with data from %s to %s.'%(first_date, last_date))

    return solar_list, first_date, last_date


def list_up_weather(weather_directory, first_date, last_date):
    # print("Current working directory in list_up_weather:", os.getcwd())
    # build weather data list
    weather_dir = os.listdir(weather_directory)
    weather_dir.sort(key=lambda x: int(x[:-1]) if x.endswith('월') else int(x))
    weather_list = []
    stridx, endidx, cnt = -1, -1, -1
    for folder in weather_dir:
        print(folder)
        wlist = os.listdir(weather_directory+'/'+folder)
        wlist = [file for file in wlist if file.find('csv') > 0]
        wlist.sort()
        for file in wlist:
            path = weather_directory + '/' + folder + '/' + file
            weather_list.append(path)
            cnt += 1
            if path.find(first_date) > 0:
                stridx = cnt
            if path.find(last_date) > 0:
                endidx = cnt

    weather_list = weather_list[stridx:endidx+1]

    return weather_list

def extract_date_from_name(filename, pre):
    return filename.split(pre)[1][:8]

def accuracy(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true+eps))) * 100

def conv_output_size(input_size, kernel_size, stride):
    return (input_size - kernel_size) // stride + 1

def pool_output_size(input_size, kernel_size, stride):
    return (input_size - kernel_size) // stride + 1

# for print
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def weights_init(m):
    """ Initialize the weights of some layers of neural networks, here Conv2D, BatchNorm, GRU, Linear
        Based on the work of Xavier Glorot
    Args:
        m: the model to initialize
    """
    classname = m.__class__.__name__
    # print("classname: ", classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('GRU') != -1:
        for weight in m.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

def insolation_aprox(t, t_rise=360, t_set=1200, I_max=2, n=2):
    if t < t_rise or t > t_set:
        return 0
    return I_max * (np.sin(np.pi * (t - t_rise) / (t_set - t_rise))) ** n

def hook_fn(module, input, output, layer_outputs:dict):
    layer_name = module.__class__.__name__
    # mean, std deviation
    mean = output.data.mean()
    std = output.data.std()
    layer_outputs[layer_name].append((mean, std))

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

def compute_percent_error(pred, y, bias):
    # Ensure inputs are tensors
    if not isinstance(y, torch.Tensor) or not isinstance(pred, torch.Tensor):
        raise ValueError("The input data is not a tensor, hence gradient descent cannot be applied. Please provide a tensor as input.")

    # Ensure inputs are 1D tensors
    if y.dim() != 1 or pred.dim() != 1:
        raise ValueError("The input data must also be a 1-dimensional tensor.")
        
    # Ensure both tensors have the same shape
    assert y.shape == pred.shape, "y and pred must have the same shape."
    
    # Compute the error
    nSamples = len(y)
    percent_error = ((torch.abs(y - pred) / (torch.abs(y) + bias)).sum())*100 / nSamples
    
    return percent_error

def monthly_mse_per_hour(y, pred, months, hours_per_day, start_month, end_month):
    mse_loss = torch.nn.MSELoss()
    start_idx = 0
    monthly_mse = []
    monthly_percent_error = []

    for month in months:
        end_idx = start_idx + month * hours_per_day
        loss = mse_loss(y[start_idx:end_idx], pred[start_idx:end_idx])
        # print("y[start_idx:end_idx].shape: ", y[start_idx:end_idx].shape)
        monthly_mse.append(loss.item())
        percent_error = compute_percent_error(pred[start_idx:end_idx], y[start_idx:end_idx], bias=0.0558) # bias = smallest_non_zero_val
        monthly_percent_error.append(percent_error.item())
        start_idx = end_idx

    for month, loss in zip(range(start_month, end_month + 1), monthly_mse):
        logging.info(f"{month} month MSE Loss: {loss:.4f} [(kW/hour)^2]")
    logging.info("")
    for month, pe in zip(range(start_month, end_month + 1), monthly_percent_error):
        logging.info(f"{month} month Percent Error: {pe:.4f} [%/hour]")
        
        # return montly_mse
