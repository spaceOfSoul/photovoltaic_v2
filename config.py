# config.py

def hyper_params():
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
        
        # after RNN layers
        "dropout": 0, 
        # correction lstm
        "previous_steps" : 5
    }

    learning_params = {
        "nBatch": 14,  # 24 hours
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
    return hparams
