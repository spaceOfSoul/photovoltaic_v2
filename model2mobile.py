import torch
from config import hyper_params
from model import RNN

model_params = hyper_params()["model"]

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

model = RNN(input_dim, output_dim, hidden_dim, rec_dropout, num_layers, 
                                      in_moving_mean, decomp_kernel, feature_wise_norm)

state_dict = torch.load('train_models/2-stageRR_2000_drop0.5_3/best_model1')
#print(state_dict['paramSet'])

model.load_state_dict(state_dict['paramSet'])

model.eval()

# example data for trace
example_input = torch.randn(1, 5, input_dim)
traced_script = torch.jit.trace(model, example_input)

traced_script.save("best_model1_mobile.pt")
