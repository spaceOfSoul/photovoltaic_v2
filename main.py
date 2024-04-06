import os
import logging
import datetime

from statsmodels.tsa.seasonal import STL
from config import hyper_params 
from ParseFlags import parse_flags 

from model import *
from utility import list_up_solar, list_up_weather
from train import train
from test import test

hparams = hyper_params()
flags, hparams, flags.model = parse_flags(hparams)

PREV_EPOCH = 500
        
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

        train(hp, flags.model, PREV_EPOCH)
        
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
        samcheok_days_per_month = [31, 28, 31, 30, 31, 30, 31, 31]  # The number of days in each month from 2022.02.01~12.31
        samcheok_start_month = 1 # 2022.01~08
        samcheok_end_month = 8    
        samcheok_filename = "samcheok_test"
        test(hp, flags.model, samcheok_days_per_month, samcheok_start_month, 
             samcheok_end_month, samcheok_filename)

        # build photovoltaic data list (GWNU_C3)
        solar_list, first_date, last_date = list_up_solar(flags.tst_gwnuC3_solar_dir)
        aws_list = list_up_weather(flags.tst_gwnuC3_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.tst_gwnuC3_asos_dir, first_date, last_date)
        
        hp.update({"loc_ID": flags.tst_gwnuC3_loc_ID})
        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})
        hp.update({"save_dir": flags.save_dir})

        
        logging.info("\n--------------------Test Mode--------------------")
        logging.info("test mode: GWNU_C3")
        gwnuC3_days_per_month = [22, 31, 31, 30, 31, 30, 31]  # The number of days in each month from 2022.02.01~12.31
        gwnuC3_start_month = 6 # 2022.06~12
        gwnuC3_end_month = 12    
        gwnuC3_filename = "gwnuC3_test"
        test(hp, flags.model, gwnuC3_days_per_month, gwnuC3_start_month, 
             gwnuC3_end_month, gwnuC3_filename)
        
    elif flags.mode == "test":
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
