import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL
import pandas as pd
import os
import sys
import torch
import time
import logging
import datetime
from config import hyper_params 
from ParseFlags import parse_flags 

from Corr import correlations
from Visualizer.LossDistribution import LossStatistics
from Visualizer.plot_generate import PlotGenerator
from Visualizer.VisualDecom import visualize_decomp
from model import *
from data_loader import WPD
from torch.utils.data import DataLoader
from utility import list_up_solar, list_up_weather, print_parameters, count_parameters, weights_init

hparams = hyper_params()
flags, hparams, flags.model = parse_flags(hparams)

def train(hparams, model_type):
    model_params = hparams["model"]
    learning_params = hparams["learning"]

    trnset = WPD(
        hparams["aws_list"],
        hparams["asos_list"],
        hparams["solar_list"],
        hparams["loc_ID"],
        input_dim=hparams["model"]["input_dim"],
    )
    valset = WPD(
        hparams["val_aws_list"],
        hparams["val_asos_list"],
        hparams["val_solar_list"],
        hparams["loc_ID"],
        input_dim=hparams["model"]["input_dim"],
    )

    trnloader = DataLoader(trnset, batch_size=1, shuffle=False, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, drop_last=True)
    
    seqLeng = model_params["seqLeng"]
    input_dim = model_params["input_dim"] # feature 7 + time 1
    output_dim = model_params["output_dim"] 
    in_moving_mean = model_params["feature_wise_norm"]
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
    lr = learning_params["lr"]   
    max_epoch = learning_params["max_epoch"]  
    
    model_classes = {"RCNN": RCNN, "RNN": RNN, "LSTM": LSTM, "correction_LSTM": correction_LSTM}

    if model_type in ["RCNN"]: 
        model = model_classes[model_type](input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          activ, cnn_dropout, kernel_size, padding, stride, nb_filters, 
                                          pooling, dropout, in_moving_mean, decomp_kernel, feature_wise_norm)      
    elif model_type in ["RNN", "LSTM"]: 
        model = model_classes[model_type](input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          in_moving_mean, decomp_kernel, feature_wise_norm)
    elif model_type in ["correction_LSTM"]:
        model = model_classes[model_type](input_dim, output_dim, hidden_dim, previous_steps, rec_dropout, num_layers, 
                                          in_moving_mean, decomp_kernel, feature_wise_norm)
    else:
        pass
    logging.info(f'The number of parameter in model : {count_parameters(model)}\n')
    model.cuda()
    model.apply(weights_init) # weights init
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    val_losses = []
    prev_loss = np.inf
    
    visualize_decomp(trnloader, period=7, seasonal=29, show=False) # show: bool
    visualize_decomp(valloader, period=7, seasonal=29, show=False) # show: bool
    train_start = time.time()
    for epoch in range(max_epoch): # epoch is current epoch
        model.train()
        loss = 0
        concat_batch_mean = torch.zeros(0).cuda()
        concat_pred = torch.zeros(0).cuda()
        concat_y = torch.zeros(0).cuda()
        prev_data = torch.zeros([seqLeng, input_dim]).cuda()

        for trn_days, (x, y) in enumerate(trnloader):
            pass
        length_trn = trn_days+1 # 365 days
                
        for trn_days, (x, y) in enumerate(trnloader):     
            #print(y.shape)       
            x = x.float()
            y = y.float()
            x = x.squeeze().cuda()
            x = torch.cat((prev_data, x), axis=0)
            prev_data = x[-seqLeng:, :]
            nLeng, nFeat = x.shape # [nLeng: 1470 min, nFeat: 8 features]           
            
            batch_data = []
            for j in range(nBatch): # nBatch: 24 hours
                stridx = j * 60 # 60 min
                endidx = j * 60 + seqLeng # seqLeng: 30 min
                batch_data.append(x[stridx:endidx, :].view(1, seqLeng, nFeat))
            batch_data = torch.cat(batch_data, dim=0) # [nBatch: 24 hours, seqLeng: 30 min, nFeat: 8 (7 features + 1 time index)]

            # print(f"trn_days:{trn_days}\nbatch_data: {batch_data}\n\n")            
            pv_power = y # [batch: 24, pv_povwer: 1]
            batch_mean = batch_data.mean(dim=1) # [batch: 24, nFeatures: 8]
           
            y = y.squeeze().cuda() # y.shape: torch.Size([24 hours])

            pred = model(batch_data.cuda()).squeeze() 

            for j in range(nBatch): # nBatch: 24 hours
                y_hourly = y[j].unsqueeze(0).cuda() # Get the hourly ground truth
                pred_hourly = pred[j].unsqueeze(0) # Get the hourly prediction
                loss_hourly = criterion(pred_hourly, y_hourly) # Calculate the hourly loss
                loss += loss_hourly.item()

            concat_batch_mean = torch.cat([concat_batch_mean, batch_mean], dim=0).cuda()
            concat_pred = torch.cat([concat_pred, pred], dim=0).cuda()
            concat_y = torch.cat([concat_y, y], dim=0).cuda()
    
            loss = loss/(length_trn * nBatch) # Calculate the average hourly loss over all days and batches
            loss_tensor = torch.tensor(loss, requires_grad=True).cuda().detach().requires_grad_(True)

            optimizer.zero_grad()
            loss_tensor.backward() # Perform backpropagation on the loss tensor
            optimizer.step()

            if epoch == 0 and trn_days == length_trn-1:                
                # pcc: Pearson Correlation Coefficient
                # ktc: Kendall's Tau Correlation Coefficient
                # Skip the first 24 entries in the tensors: 맨 처음 24시간은 (초기값) zero-padding이므로
                concat_y_us = concat_y.unsqueeze(-1)[24:] # [8736, 1] after skipping the first 24
                concat_batch_mean = concat_batch_mean[24:] # [8736, 8] after skipping the first 24
                pv_pcc, features_pcc, pv_ktc, features_ktc = correlations(concat_y_us, concat_batch_mean, plot=hparams["plot_corr"])
 
        ######### Validation ######### 
        model.eval()
        val_loss = 0
        val_pred_append = []
        val_y_append=[]  
        concat_pred_val = torch.zeros(0).cuda()
        concat_y_val = torch.zeros(0).cuda()    
        prev_data = torch.zeros([seqLeng, input_dim]).cuda()
        
        for val_days, _ in enumerate(valloader):
            pass
        length_val = val_days+1 # 365 days: 어린이집(PreSchool)
        
        for val_days, (x, y) in enumerate(valloader):            
            x = x.float()
            y = y.float()
            x = x.squeeze().cuda()
            x = torch.cat((prev_data, x), axis=0).cuda()
            prev_data = x[-seqLeng:, :]
            y = y.squeeze().cuda()
            nLeng, nFeat = x.shape
            
            batch_data = []
            
            for j in range(nBatch):
                stridx = j * 60
                endidx = j * 60 + seqLeng
                batch_data.append(x[stridx:endidx, :].view(1, seqLeng, nFeat))               
            batch_data = torch.cat(batch_data, dim=0) # concatenate along the batch dim
            pred = model(batch_data.cuda()).squeeze()
            val_loss += criterion(pred, y)

            concat_pred_val = torch.cat([concat_pred_val, pred], dim=0).cuda()
            concat_y_val = torch.cat([concat_y_val, y], dim=0).cuda()

            val_pred_append.append(pred.detach().cpu().numpy())
            val_y_append.append(y.detach().cpu().numpy())
                                                           
        if val_loss < prev_loss:
            savePath = os.path.join(hparams["save_dir"], "best_model")  # overwrite
            model_dict = {"kwargs": model_params, "paramSet": model.state_dict()}
            torch.save(model_dict, savePath)
            prev_loss = val_loss
            val_BestPred = val_pred_append
        
        val_loss = val_loss/(length_val) # val_loss = (1/(24*length_val)) * Σ(y_i - ŷ_i)^2 (i: from 1 to 24*length_val) [(kW/h)^2]
        
        loss_trn = loss_tensor.item() 
        losses.append(loss_trn)
        loss_val = val_loss.item()
        val_losses.append(loss_val)

        logging.info(f"Epoch [{epoch+1}/{max_epoch}], Trn Loss: {loss_trn:.4f}, Val Loss: {loss_val:.4f} [(kW/hour)^2]")

    train_end = time.time()
    logging.info("\n")
    logging.info(f'Training time [sec]: {(train_end - train_start):.2f}\n')
    logging.info(f'len(Trn_Loss): {len(losses)}\nTrn Loss [(kW/hour)^2]: \n{" ".join([f"{loss:.4f}" for loss in losses])}')


    min_val_loss = min(val_losses)
    logging.info(f'\nlen(Val_Loss): {len(val_losses)}\nVal Loss [(kW/hour)^2]: \n{" ".join([f"{loss:.4f}" for loss in val_losses])}')
    logging.info(f'Min validation loss : {min_val_loss}')
    min_val_loss_epoch = val_losses.index(min_val_loss)
    
    # plot origin, trend, seasonal, residual of Validating  
    image_dir = os.path.join(hparams["save_dir"], "best_val_images")
    os.makedirs(image_dir, exist_ok=True)
    
    days_per_month = [28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # ㅔchange to hp
    start_month = 2 # 2022.01~12

    val_y_chunks = []
    val_BestPred_chunks = []

    start_index = 0
    for month_length in days_per_month:
        end_index = start_index + month_length
        val_y_chunks.append(val_y_append[start_index:end_index])
        val_BestPred_chunks.append(val_BestPred[start_index:end_index])
        start_index = end_index
    
    plot_generator = PlotGenerator(image_dir, days_per_month, start_month)
    plot_generator.plot_monthly(val_y_chunks, val_BestPred_chunks)
    plot_generator.plot_annual(val_y_append, val_BestPred)
    plot_generator.plot_monthly_loss(val_y_append, val_BestPred)
     ##############
    if hparams["loss_plot_flag"]:
    
        loss_stats = LossStatistics(losses, val_losses)
        loss_stats.process(hparams["save_dir"])

        plt.figure()
        plt.plot(range(max_epoch), np.array(losses), "b", label='Training Loss')
        plt.plot(range(max_epoch), np.array(val_losses), "r", label='Validation Loss')
        min_val_loss = min(val_losses)
        min_val_loss_epoch = val_losses.index(min_val_loss)
        plt.scatter(min_val_loss_epoch, min_val_loss, color='k', label='Minimun Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss, Min Val Loss: {min_val_loss:.4f} at Epoch {min_val_loss_epoch}")
        plt.legend()
        plt.savefig(os.path.join(hparams["save_dir"],"figure_train.png"))
        logging.info(f"minimum validation loss: {min_val_loss:.4f} [(kW/hour)^2] at Epoch {min_val_loss_epoch}")
        
def test(hparams, model_type):
    model_params = hparams['model']
    learning_params = hparams['learning']

    modelPath = hparams['load_path']
    
    try:
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
    in_moving_mean = model_params["feature_wise_norm"]
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
    lr = learning_params["lr"]   
    max_epoch = learning_params["max_epoch"]  
    
    model_classes = {"RCNN": RCNN, "RNN": RNN, "LSTM": LSTM, "correction_LSTM": correction_LSTM}

    if model_type in ["RCNN"]: 
        model = model_classes[model_type](input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          activ, cnn_dropout, kernel_size, padding, stride, nb_filters, 
                                          pooling, dropout, in_moving_mean, decomp_kernel, feature_wise_norm)      
    elif model_type in ["RNN", "LSTM"]: 
        model = model_classes[model_type](input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                          in_moving_mean, decomp_kernel, feature_wise_norm)  
    elif model_type in ["correction_LSTM"]:
           model = model_classes[model_type](input_dim, output_dim, hidden_dim, previous_steps, rec_dropout, num_layers, 
                                         in_moving_mean, decomp_kernel, feature_wise_norm)    
    else:
        pass
        
    model.load_state_dict(paramSet)
    model.cuda()
    model.eval()

    tstset = WPD(hparams['aws_list'], hparams['asos_list'], hparams['solar_list'], hparams['loc_ID'])    
    tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)
    visualize_decomp(tstloader, period=7, seasonal=29, show=False) # show: bool
    
    prev_data = torch.zeros([seqLeng, input_dim]).cuda()	# 7 is for featDim

    criterion = torch.nn.MSELoss()
    total_mape = 0
    total_samples = 0
    result = []
    y_true=[]
    test_start = time.time()

    tst_loss = 0
    
    for tst_days, (x, y) in enumerate(tstloader):
        pass
    length_tst = tst_days+1        
    
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
        pred = model(batch_data).squeeze() # torch.Size([24])
          
        result.append(pred.detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())
    
        tst_loss += criterion(pred, y)
    
    test_end = time.time()
    
    tst_loss = tst_loss/length_tst # tst_loss = (1/(24*length_tst)) * Σ(y_i - ŷ_i)^2 (i: from 1 to 24*length_tst) [(kW/h)^2]
    loss_tst = tst_loss.item()

    # print_parameters(model) # print params infomation
    logging.info(f'Test Loss: {loss_tst:.4f} [(kW/hour)^2]')
    logging.info(f'The number of parameter in model : {count_parameters(model)}')
    logging.info(f'Testing time [sec]: {(test_end - test_start):.2f}')

    model_dir = os.path.dirname(modelPath)
    
    if hparams['save_result']:
        result_npy = np.array(result)
        np.save(os.path.join(model_dir,"test_result.npy"), result_npy)

    test_dir = os.path.join(model_dir, f"tests_{hparams['loc_ID']}")
    os.makedirs(test_dir, exist_ok=True)

    result_arr = np.concatenate(result, axis=0)
    y_true_arr = np.concatenate(y_true, axis=0)


    days = 30
    hour_chunk = days * 24

    num_chunks = len(result_arr) // hour_chunk

    for i in range(num_chunks):
        start_idx = i * hour_chunk
        end_idx = start_idx + hour_chunk

        plt.figure()
        plt.plot(result_arr[start_idx:end_idx].flatten(), label='Predictions')
        plt.plot(y_true_arr[start_idx:end_idx].flatten(), label='Ground Truth')
        plt.xlabel('Time (hours)')
        plt.ylabel('Output')
        plt.legend()
        plt.title(f'Hourly predictions (Day {i*days+1} to Day {(i+1)*days})')
        plt.savefig(os.path.join(test_dir, f'predictions{i*days+1}_to_{(i+1)*days}.png'))
        plt.close()

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
    logging.info('--------------------Hyperparameters--------------------\n')
    for key, value in hp.items():
        logging.info(f"{key}: {value}\n")
    logging.info(f"Model name: {model_name}\n")
    
    if flags.mode == "train":
        logging.info("\n--------------------Training Mode (Training and Validating)--------------------")
        # =============================== training data list ====================================#
        # build photovoltaic data list
        solar_list, first_date, last_date = list_up_solar(flags.solar_dir)
        aws_list = list_up_weather(flags.aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.asos_dir, first_date, last_date)
        logging.info(f"Training on the interval from {first_date} to {last_date}.")
        # =============================== validation data list ===================================#
        # build photovoltaic data list
        val_solar_list, first_date, last_date = list_up_solar(flags.val_solar_dir)
        # print(f"first_date: {first_date}, last_date: {last_date}")
        # first_date = "20220101"
        val_aws_list = list_up_weather(flags.val_aws_dir, first_date, last_date)
        val_asos_list = list_up_weather(flags.val_asos_dir, first_date, last_date)
        logging.info(f"Validating on the interval from {first_date} to {last_date}.\n")
        # ========================================================================================#

        hp.update({"aws_list": aws_list})
        hp.update({"val_aws_list": val_aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"val_asos_list": val_asos_list})
        hp.update({"solar_list": solar_list})
        hp.update({"val_solar_list": val_solar_list})
        hp.update({"save_dir": flags.save_dir})
        hp.update({"loc_ID": flags.loc_ID})

        train(hp, flags.model)
        
        hp.update({"load_path": os.path.join(flags.save_dir,"best_model")})
        
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
        test(hp, flags.model)

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
        test(hp, flags.model)
        
    elif flags.mode == "test":
        hp.update({"load_path": flags.load_path})
        hp.update({"loc_ID": flags.tst_samcheok_loc_ID})
        
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
        test(hp, flags.model)

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
        test(hp, flags.model)

        solar_list, first_date, last_date = list_up_solar(flags.val_solar_dir)
        aws_list = list_up_weather(flags.val_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.val_asos_dir, first_date, last_date)
        
        hp.update({"loc_ID": 678})
        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})

        logging.info("\n--------------------Validation Mode--------------------")
        logging.info("test mode: GWNU_Preschool")
        test(hp, flags.model)