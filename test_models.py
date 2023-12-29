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

def load_model(model_path):
    # 모델을 로드하는 함수입니다. 모델 구조에 따라 다르게 적용해야 합니다.
    model = torch.load(model_path)
    model.eval()
    return model

def plot_predictions(ground_truth, model_predictions, labels):
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth, label='Ground Truth', color='black', linewidth=2)
    for i, prediction in enumerate(model_predictions):
        plt.plot(prediction, label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Model Predictions vs Ground Truth')
    plt.legend()
    plt.show()

def test_multiple_models(model_paths, test_data_loader):
    models = [load_model(path) for path in model_paths]
    model_predictions = []
    ground_truth = None

    for data in test_data_loader:
        x, y = data
        x = x.cuda()  # 데이터가 GPU에서 실행되는 경우
        if ground_truth is None:
            ground_truth = y.numpy()

        for model in models:
            with torch.no_grad():
                pred = model(x).cpu().numpy()
                model_predictions.append(pred)
    
    # 각 모델에 대한 라벨 생성
    labels = [f'Model {i+1}' for i in range(len(models))]
    
    plot_predictions(ground_truth, model_predictions, labels)

# 모델 경로 배열
model_paths = ['model1.pth', 'model2.pth', 'model3.pth', ...]

# 테스트 데이터 로더 준비
# test_dataset = YourDataset(...)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 함수 호출
# test_multiple_models(model_paths, test_loader)
