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

def test(hparams, model_type, days_per_month, start_month, end_month, filename):
    model_params = hparams['model']
    learning_params = hparams['learning']

    modelPath = hparams['load_path']
    
    try:
        if model_type in ["correction_LSTMs", "2-stageLR", "2-stageRL", "2-stageRR"]:
            model1Path = os.path.join(os.path.dirname(modelPath), "best_model1")
            model2Path = os.path.join(os.path.dirname(modelPath), "best_model2")

            ckpt1 = torch.load(model1Path)
            ckpt2 = torch.load(model2Path)

            if not isinstance(ckpt1, dict):
                raise ValueError(f"Loaded object from {modelPath} is not a dictionary.")
            if 'kwargs' not in ckpt1 or 'paramSet' not in ckpt1:
                raise ValueError(f"Dictionary from {modelPath} does not contain expected keys.")

            model_conf1 = ckpt1['kwargs']
            paramSet1 = ckpt1['paramSet']

            model_conf2 = ckpt2['kwargs']
            paramSet2 = ckpt2['paramSet']

        else:
            ckpt = torch.load(modelPath)
            if not isinstance(ckpt, dict):
                raise ValueError(f"Loaded object from {modelPath} is not a dictionary.")
            if 'kwargs' not in ckpt or 'paramSet' not in ckpt:
                raise ValueError(f"Dictionary from {modelPath} does not contain expected keys.")
            model_conf = ckpt['kwargs']
            paramSet = ckpt['paramSet']
    except Exception as e:
        print(f"Error occurred while loading model from {modelPath}")
        print(f"Error: {e}")
    
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
    
    model_classes = {"RCNN": RCNN, "RNN": RNN, "LSTM": LSTM,
                     "correction_LSTMs": (LSTM,LSTM), "2-stageLR":(LSTM,RNN), "2-stageRL":(RNN,LSTM), "2-stageRR":(RNN,RNN)}
    
    if model_type in ["RCNN"]: 
        model = model_classes[model_type](input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          activ, cnn_dropout, kernel_size, padding, stride, nb_filters, 
                                          pooling, dropout, in_moving_mean, decomp_kernel, feature_wise_norm)      
    elif model_type in ["RNN", "LSTM"]: 
        model = model_classes[model_type](input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          in_moving_mean, decomp_kernel, feature_wise_norm)
    elif model_type in ["correction_LSTMs", "2-stageLR", "2-stageRL", "2-stageRR"]:
        model1 = model_classes[model_type][0](input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          in_moving_mean, decomp_kernel, feature_wise_norm)
        model2 = model_classes[model_type][1](previous_steps, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          in_moving_mean, decomp_kernel, feature_wise_norm)
        prev_preds = deque(maxlen=previous_steps)
        val_prev_preds = deque(maxlen=previous_steps)
    else:
        pass
        
    if model_type in ["correction_LSTMs", "2-stageLR", "2-stageRL", "2-stageRR"]:
        model1.load_state_dict(paramSet1)
        model2.load_state_dict(paramSet2)   
        model1.cuda().eval()
        model2.cuda().eval()
    else:
        model.load_state_dict(paramSet)
        model.cuda()
        model.eval()
    
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
    
    for tst_days, (x, y) in enumerate(tstloader):
        x = x.float()
        y = y.float()
        x = x.squeeze().cuda()   
                                 
        x = torch.cat((prev_data, x), axis=0)
        prev_data = x[-seqLeng:, :]
        y = y.squeeze().cuda()
        nLeng, nFeat = x.shape
        
        batch_data = []
            
        for j in range(nBatch):
            stridx = j * 60
            endidx = j * 60 + seqLeng
            batch_data.append(x[stridx:endidx, :].view(1, seqLeng, nFeat))
             
        batch_data = torch.cat(batch_data, dim=0) # concatenate along the batch dim
        if model_type in ["correction_LSTMs", "2-stageLR", "2-stageRL", "2-stageRR"]:
            first_pred= model1(batch_data.cuda()).squeeze()
                
            #if epoch >PREV_EPOCH:
            prev_preds.append(first_pred.detach().clone())
            while len(prev_preds) < previous_steps:
                prev_preds.appendleft(torch.zeros_like(first_pred))
            final_input = torch.stack(list(prev_preds), dim=1)
            pred= model2(final_input).squeeze() 
        else:
            pred = model(batch_data).squeeze() # torch.Size([24])

        batch_data_daytime = batch_data[batch_start_idx:batch_end_idx, :, :] # assume: batch_start_idx: 5, batch_end_idx: 21; 0~4, 21~23시의 데이터는 제거             
        pred_daytime = pred[batch_start_idx:batch_end_idx].squeeze()
        y_daytime = y[batch_start_idx:batch_end_idx]   

        #print(f"pred : {pred.shape} y : {y.shape}")
        tst_loss_alltime += criterion(pred.squeeze(), y)
        tst_loss_daytime += criterion(pred_daytime, y_daytime)
        percent_error_tst_alltime += compute_percent_error(pred.squeeze(), y, bias=0.0558) # bias = smallest_non_zero_val
        percent_error_tst_daytime += compute_percent_error(pred_daytime, y_daytime, bias=0.0558) # bias = smallest_non_zero_val

        concat_pred_tst_alltime = torch.cat([concat_pred_tst_alltime, pred.squeeze()], dim=0).cuda()
        concat_y_tst_alltime = torch.cat([concat_y_tst_alltime, y], dim=0).cuda()
        concat_pred_tst_daytime = torch.cat([concat_pred_tst_daytime, pred_daytime], dim=0).cuda()
        concat_y_tst_daytime = torch.cat([concat_y_tst_daytime, y_daytime], dim=0).cuda()
        
        tst_pred_append.append(pred.detach().cpu().numpy())
        tst_y_append.append(y.detach().cpu().numpy())
    
    tst_loss_alltime = tst_loss_alltime/(length_tst) # tst_loss_alltime = (1/(24*length_tst)) * Σ(y_i - ŷ_i)^2 (i: from 1 to 24*length_tst) [(kW/h)^2]
    tst_loss_daytime = tst_loss_daytime/(length_tst) # tst_loss_daytime = (1/(daytime_hours*length_tst)) * Σ(y_i - ŷ_i)^2 (i: from 1 to daytime_hours*length_tst) [(kW/h)^2]
    percent_error_tst_alltime = percent_error_tst_alltime/(length_tst) # percent_error_alltime = (1/(alltime_hours*length_tst)) * Σ(abs((y_i - ŷ_i))/(y_i)) * 100 [%/hour]
    percent_error_tst_daytime = percent_error_tst_daytime/(length_tst) # percent_error_daytime = (1/(daytime_hours*length_tst)) * Σ(abs((y_i - ŷ_i))/(y_i)) * 100 [%/hour]
        
    loss_tst_alltime = tst_loss_alltime.item() 
    tst_losses_alltime.append(loss_tst_alltime)
    tst_loss_daytime = tst_loss_daytime.item()
    tst_losses_daytime.append(tst_loss_daytime)

    percent_error_tst_alltime = percent_error_tst_alltime.item()
    percent_errors_tst_alltime.append(percent_error_tst_alltime)         
    percent_error_tst_daytime = percent_error_tst_daytime.item()
    percent_errors_tst_daytime.append(percent_error_tst_daytime)        
        
    logging.info(f"tst Loss_alltime: {loss_tst_alltime:.4f} [(kW/hour)^2]")
    logging.info(f"tst Daytime Loss: {tst_loss_daytime:.4f} [(kW/hour)^2]")
    logging.info(f"tst % alltime error:{percent_error_tst_alltime:.2f} [%/hour]")
    logging.info(f"tst % daytime error:{percent_error_tst_daytime:.2f} [%/hour]\n")

    ################################################################################
    # plot origin, trend, seasonal, residual of Test  
    image_dir = os.path.join(hparams["save_dir"], filename)
    os.makedirs(image_dir, exist_ok=True)

    test_y_chunks = []
    test_pred_chunks = []

    start_index = 0
    for month_length in days_per_month:
        end_index = start_index + month_length
        test_y_chunks.append(tst_y_append[start_index:end_index])
        test_pred_chunks.append(tst_pred_append[start_index:end_index])
        start_index = end_index
    logging.info("################################################################")
    plot_generator = PlotGenerator(image_dir, days_per_month, start_month)
    plot_generator.plot_monthly(test_y_chunks, test_pred_chunks)
    logging.info(f"{filename} results")
    plot_generator.plot_annual(tst_y_append, tst_pred_append)
    month_label = [str(i) for i in range(start_month, end_month + 1)]
    plot_generator.plot_monthly_loss(tst_y_append, tst_pred_append, month_label)

    # Filter out zeros and find the minimum of the remaining values
    smallest_non_zero_test = concat_y_tst_daytime[concat_y_tst_daytime > 0].min().item()
    # print(smallest_non_zero_test) # samcheok: 0.09, GWNU_C3: 0.01 [kW/h]
       
    logging.info("################################################################")
    logging.info("\n# Alltime Info Of Test\nMSE loss(alltime) and Percent Error(alltime) of (trend, seasonal, residual)")
    compute_mse_and_errors(pred=concat_pred_tst_alltime, y=concat_y_tst_alltime, period=alltime_hours, seasonal=((alltime_hours*30)-1), bias=smallest_non_zero_test)
    monthly_mse_per_hour(concat_y_tst_alltime, concat_pred_tst_alltime, days_per_month, alltime_hours, start_month, end_month)

    logging.info("\n################################################################")
    logging.info("\n# Daytime Info Of Test\nMSE loss(daytime) and Percent Error(daytime) of (trend, seasonal, residual)")
    compute_mse_and_errors(pred=concat_pred_tst_daytime, y=concat_y_tst_daytime, period=daytime_hours, seasonal=((daytime_hours*30)-1), bias=smallest_non_zero_test)
    monthly_mse_per_hour(concat_y_tst_daytime, concat_pred_tst_daytime, days_per_month, daytime_hours, start_month, end_month)
    
    ################################################################################
    # print_parameters(model) # print params infomation
    test_end = time.time()
    if  model_type in ["correction_LSTMs", "2-stageLR", "2-stageRL", "2-stageRR"]:
        logging.info(f'The number of parameter in model1 : {count_parameters(model1)}')
        logging.info(f'The number of parameter in model2 : {count_parameters(model2)}')
    else:
        logging.info(f'The number of parameter in model : {count_parameters(model)}')
    logging.info(f'Testing time [sec]: {(test_end - test_start):.2f}')

    model_dir = os.path.dirname(modelPath)
    
    if hparams['save_result']:
        tst_pred_npy = np.array(tst_pred_append)
        np.save(os.path.join(model_dir,"test_pred.npy"), tst_pred_npy)
