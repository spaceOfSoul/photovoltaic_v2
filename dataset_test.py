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

aws_dir="./dataset/AWS/"
asos_dir="./dataset/ASOS/"
solar_dir = "./dataset/photovoltaic/GWNU_C9/"

solar_list, first_date, last_date = list_up_solar(solar_dir)
aws_list = list_up_weather(aws_dir, first_date, last_date)
asos_list = list_up_weather(asos_dir, first_date, last_date)
isol_list = ["samcheok/pv_f16.csv"]

dataset = WPD(
        aws_list,
        asos_list,
        solar_list,
        isol_list,
        (first_date, last_date),
        105
    )

for _,data in enumerate(dataset):
    print(data.shape)