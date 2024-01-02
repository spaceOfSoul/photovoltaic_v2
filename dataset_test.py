

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

hparams = hyper_params()
flags, hparams, flags.model = parse_flags(hparams)

PREV_EPOCH = 500

def test(hparams, model_type, days_per_month, start_month, end_month, filename):
    model_params = hparams['model']
    learning_params = hparams['learning']

    modelPath = hparams['load_path']
    
    seqLeng = model_params["seqLeng"]
    input_dim = model_params["input_dim"] # feature 7 + time 1
    output_dim = model_params["output_dim"]

    #apply_kernel_func = model_params["apply_kernel_func"] 
    #kernel_start_idx = model_params["kernel_start_idx"]
    #kernel_end_idx = model_params["kernel_end_idx"]
    #kernel_center = model_params["kernel_center"]
    #kernel_feature_idx = model_params["kernel_feature_idx"]
    #kernel_type = model_params["kernel_type"]

    batch_start_idx = model_params["batch_start_idx"]
    batch_end_idx = model_params["batch_end_idx"]

    in_moving_mean = model_params["in_moving_mean"]
    decomp_kernel = model_params["decomp_kernel"]
    feature_wise_norm = model_params["feature_wise_norm"]
           
    hidden_dim = model_params["nHidden"]
    rec_dropout = model_params["rec_dropout"]
    num_layers = model_params["num_layers"]
    activ = model_params["activ"]
    cnn_dropout = model_params["cnn_dropout"]  
    kernel_size = model_params["kernel_size"]
    padding = model_params["padding"]
    stride = model_params["stride"]
    nb_filters = model_params["nb_filters"]
    pooling = model_params["pooling"]
    dropout = model_params["dropout"]
           
    previous_steps = model_params["previous_steps"]
     
    nBatch = learning_params["nBatch"]
    #lr = learning_params["lr"]   
    #max_epoch = learning_params["max_epoch"] 
    
    
    tstset = WPD(hparams['aws_list'], hparams['asos_list'], hparams['solar_list'], hparams['loc_ID'], input_dim=hparams["model"]["input_dim"],)    
    tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)
    visualize_decomp(tstloader, period=7, seasonal=29, show=False) # show: bool
    
    prev_data = torch.zeros([seqLeng, input_dim]).cuda()	# 7 is for featDim

    criterion = torch.nn.MSELoss()

    tst_losses_alltime = []
    tst_losses_daytime = []
    percent_errors_tst_alltime = []
    percent_errors_tst_daytime = []

    tst_loss_alltime = 0
    tst_loss_daytime = 0
    percent_error_tst_alltime = 0
    percent_error_tst_daytime = 0
    tst_pred_append = []
    tst_y_append=[] 
    concat_pred_tst_alltime = torch.zeros(0).cuda()
    concat_y_tst_alltime = torch.zeros(0).cuda()     
    concat_pred_tst_daytime = torch.zeros(0).cuda()
    concat_y_tst_daytime = torch.zeros(0).cuda()  
    alltime_hours = 24 # 24 hours
    daytime_hours = (batch_end_idx - batch_start_idx -1)    

    test_start = time.time()

    tst_loss = 0
    
    for tst_days, (x, y) in enumerate(tstloader):
        print(f'{x.shape} {y.shape}')
    length_tst = tst_days+1
    print(f"length : {length_tst}")
    
    for tst_days, (x, y, z) in enumerate(tstloader):
        x = x.float()
        y = y.float()
        x = x.squeeze().cuda()   
                                 
        logging.info(x.shape)
        logging.info(y.shape)
        logging.info(z.shape)

    

if __name__ == "__main__":

    hp = hyper_params()
    flags, hp, model_name = parse_flags(hp)

    # python main.py --mode train --model RCNN --save_dir temp_train --log_filename text_file.txt

    # python main.py --mode test --model RCNN --load_path temp_train/best_model --save_dir temp_test --log_filename text_file.txt
 
    # Set up logging
    if not os.path.isdir(flags.save_dir):
        os.makedirs(flags.save_dir)
        
    log_filename = os.path.join(flags.save_dir, flags.log_filename)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    # Then replace 'print' with 'logging.info' in your code   
    logging.info('\n###################################################################')
    logging.info('###################################################################')
    logging.info('###################################################################\n')
    current_time = datetime.datetime.now()
    logging.info(f"Current time: {current_time}\n")

    # Log hyperparameters and model name
    logging.info('--------------------dataloader test--------------------\n')

    if flags.mode == "test":
        hp.update({"load_path": flags.load_path})
        hp.update({"loc_ID": flags.tst_samcheok_loc_ID})
        hp.update({"save_dir": flags.save_dir})
        
        # =============================== test data list ====================================#
        # build photovoltaic data list (samcheok)
        solar_list, first_date, last_date = list_up_solar(flags.tst_samcheok_solar_dir)
        aws_list = list_up_weather(flags.tst_samcheok_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.tst_samcheok_asos_dir, first_date, last_date)
        
        hp.update({"loc_ID": flags.tst_samcheok_loc_ID})        
        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})
        
        logging.info("\n--------------------Test Mode--------------------")        
        logging.info("test mode: samcheok")

        samcheok_days_per_month = [31, 28, 31, 30, 31, 30, 31, 31]
        samcheok_start_month = 1 # 2022.01~08
        samcheok_end_month = 8    
        samcheok_filename = "samcheok_test"   
        test(hp, flags.model, samcheok_days_per_month, samcheok_start_month, samcheok_end_month, samcheok_filename)

        # build photovoltaic data list (GWNU_C3)
        solar_list, first_date, last_date = list_up_solar(flags.tst_gwnuC3_solar_dir)
        aws_list = list_up_weather(flags.tst_gwnuC3_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.tst_gwnuC3_asos_dir, first_date, last_date)
        
        hp.update({"loc_ID": flags.tst_gwnuC3_loc_ID})
        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})
        
        logging.info("\n--------------------Test Mode--------------------")
        logging.info("test mode: GWNU_C3")
        gwnuC3_days_per_month = [22, 31, 31, 30, 31, 30, 31]
        gwnuC3_start_month = 6 # 2022.06~12
        gwnuC3_end_month = 12    
        gwnuC3_filename = "gwnuC3_test"
        test(hp, flags.model, gwnuC3_days_per_month, gwnuC3_start_month, 
             gwnuC3_end_month, gwnuC3_filename)

        solar_list, first_date, last_date = list_up_solar(flags.val_solar_dir)
        aws_list = list_up_weather(flags.val_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.val_asos_dir, first_date, last_date)
        
        hp.update({"loc_ID": 678})
        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})

        logging.info("\n--------------------Validation Mode--------------------")
        logging.info("test mode: GWNU_Preschool")
        gwnuPreSch_days_per_month = [28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # The number of days in each month from 2022.02.01~12.31
        gwnuPreSch_start_month = 2 # 2022.02~12
        gwnuPreSch_end_month = 12    
        gwnuPreSch_filename = "GWNU_Preschool"
        test(hp, flags.model, gwnuPreSch_days_per_month, gwnuPreSch_start_month, 
             gwnuPreSch_end_month, gwnuPreSch_filename)
