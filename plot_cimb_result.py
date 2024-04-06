import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torch
import time
import logging
import datetime
from collections import deque

from statsmodels.tsa.seasonal import STL
from config import hyper_params 
from ParseFlags import parse_flags 

from Corr import correlations
from Visualizer.LossDistribution import LossStatistics
from Visualizer.plot_generate import PlotGenerator
from Visualizer.VisualDecom import visualize_decomp
from model import *
from data_loader import WPD
from torch.utils.data import DataLoader
from utility import list_up_solar, list_up_weather, print_parameters, count_parameters, weights_init, compute_percent_error, monthly_mse_per_hour, compute_mse_and_errors
from functools import partial

# Default setting
nlayers = 2  # nlayers of CNN 
model_params = {          
    # Common
    "seqLeng": 60,  # [min] # 50, 40, 30, 20, 10 으로 해보기
    "input_dim": 8,  # feature 7 + time 1
    "output_dim": 1,  # PV power
    
    # Preprocessing
    "in_moving_mean": False, # inputs = series_decomp(inputs)
    "decomp_kernel": [3, 6, 12], # kernel size of series_decomp
    "feature_wise_norm": False,  # bool (True or False); normalize input feature
            
    # RNN
    "nHidden": 128, 
    "rec_dropout": 0, 
    "num_layers": 2,
                    
    # CNN
    "activ": "relu", # leakyrelu, relu, glu, cg
    "cnn_dropout": 0, 
    "kernel_size": nlayers*[3],
    "padding": nlayers*[1],
    "stride": nlayers*[1], 
    "nb_filters": [16, 32], # length of nb_filters should be equal to nlayers.
    "pooling": nlayers*[1],   
    
    "batch_start_idx": 5, # 0~(batch_start_idx-1) 시간대는 제외
    "batch_end_idx": 21, # batch_end_idx~23 시간대는 제외
    # after RNN layers
    "dropout": 0.5, 
    # correction lstm
    "previous_steps" : 5
}
learning_params = {
    "nBatch": 24,  # 24 hours
    "lr": 1.0e-3,
    "max_epoch": 2000,
}
hparams = {
    "model": model_params,
    "learning": learning_params,
    # system flags
    "plot_corr": False, # Pearson Correlation Coefficient, Kendall's Tau Correlation Coefficient
    "loss_plot_flag": True,
    "save_losses": True,
    "save_result": True,
}

npy_files = [
    'train_models/2-stageLR_2000_drop0.5_3/test_pred.npy',
    'train_models/2-stageRR_2000_drop0.5_3/test_pred.npy', 
    'train_models/2-stageRL_2000_drop0.5_3/test_pred.npy', 
    'train_models/2-stageLL_2000_drop0.5_3/test_pred.npy', 
    'train_models/LSTM_2000_drop0.5_3/test_pred.npy',
    'train_models/RNN_2000_drop0.5_3/test_pred.npy'
    ] 
npy_daily_data = [[] for _ in npy_files]

for i, file in enumerate(npy_files):
    data = np.load(file)
    for j in range(0, len(data)):
        npy_daily_data[i].append(data[j])

solar_list, first_date, last_date = list_up_solar('./dataset/photovoltaic/GWNU_C9/')
aws_list = list_up_weather('./dataset/AWS/', first_date, last_date)
asos_list = list_up_weather('./dataset/ASOS/', first_date, last_date)

tstset = WPD(aws_list, asos_list, solar_list, 678, input_dim=hparams["model"]["input_dim"],)    
tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)
daily_data = []
# npy 파일들에서 데이터 로딩 및 일별 데이터 준비
for i, file in enumerate(npy_files):
    data = np.load(file)
    print(f"Shape of data from {file}: {data.shape}")  # npy 파일 데이터 shape 확인
    for j in range(0, len(data)):
        npy_daily_data[i].append(data[j])

# tstloader에서 데이터 로딩
daily_data = []
for _, day_data in tstloader:
    day_data_squeezed = day_data.squeeze().numpy()
    print(f"Shape of daily data from tstloader: {day_data_squeezed.shape}")  # tstloader 데이터 shape 확인
    daily_data.append(day_data_squeezed)

# 플롯 그리기
num_days = min(len(daily_data), min(len(data) for data in npy_daily_data))
# 하루에 대한 모든 모델의 예측과 실제 데이터를 한 그래프에 표시
selected_day = 0  # 예를 들어, 첫째 날
# 한 날짜의 데이터를 별도의 plot으로 생성하여 저장
for day in range(len(npy_daily_data[0])):  # 예를 들어 334일에 대해 반복
    plt.figure(figsize=(10, 6))

    # 각 npy 파일에서의 예측값을 plot
    for i, dataset in enumerate(npy_daily_data):
        dir_path = os.path.dirname(npy_files[i])
        folder_name = os.path.basename(dir_path)
        plt.plot(dataset[day], label=f'{folder_name} Prediction')

    # tstloader에서의 실제 발전량을 plot (하루에 대한 데이터는 모두 동일)
    if day < len(daily_data):  # tstloader 데이터가 충분한 경우에만 plot
        plt.plot(daily_data[day], label='Actual Power', linestyle='dashed')

    plt.title(f'Predictions vs Actual Power for Day {day + 1}')
    plt.xlabel('Hour')
    plt.ylabel('Power')
    plt.legend()

    # plot 저장
    plt.savefig(f'test_plot/plot_day_{day + 1}.png')
    plt.close()
