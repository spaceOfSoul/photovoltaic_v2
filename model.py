import torch
from torch import nn
from models.RNNs import RNNModule, LSTMModule
from models.CNN import CNNModule
from models.SeriesDecomp import series_decomp_multi
        
class RCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, rec_dropout=0, num_layers=1, activ="Relu", cnn_dropout=0, kernel_size=2*[3], padding=2*[1], stride=2*[1], nb_filters=[64, 128], pooling=2*[1], dropout=0, in_moving_mean=True, decomp_kernel=[3, 5, 7, 9], feature_wise_norm=True):
        super(RCNN, self).__init__()

        self.feature_wise_norm = feature_wise_norm
        self.in_moving_mean = in_moving_mean
        self.decomp_kernel = decomp_kernel
        self.series_decomp_multi = series_decomp_multi(kernel_size=self.decomp_kernel)

        self.lstm = LSTMModule(input_dim=input_dim, hidden_dim=hidden_dim, rec_dropout=rec_dropout, num_layers=num_layers)          
        self.cnn = CNNModule(n_in_channel=hidden_dim, activ=activ, conv_dropout=cnn_dropout, kernel_size=kernel_size, padding=padding, stride=stride, nb_filters=nb_filters, pooling=pooling)
        self.dropout = nn.Dropout(dropout)   
            
        self.dense = nn.Linear(nb_filters[-1], output_dim)
        # self.sigmoid = nn.Sigmoid() # 회귀문제에서는 분류문제와 다르게 출력값을 시그모이드로 0과 1사이의 값으로 제한하면 출력 범위가 제한되므로 사용하면 안 된다.
        self.dense_softmax = nn.Linear(nb_filters[-1], output_dim)
        self.softmax = nn.Softmax(dim=-1) 

    def load_lstm(self, state_dict):
        self.lstm.load_state_dict(state_dict)
        
    def load_state_dict(self, state_dict, strict=True):
        self.series_decomp_multi.load_state_dict(state_dict["series_decomp_multi"])
        self.lstm.load_state_dict(state_dict["lstm"])
        self.cnn.load_state_dict(state_dict["cnn"])
        self.dense.load_state_dict(state_dict["dense"])
        self.dense_softmax.load_state_dict(state_dict["dense_softmax"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"series_decomp_multi": self.series_decomp_multi.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "lstm": self.lstm.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars), 
                      "cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense": self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense_softmax": self.dense_softmax.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'series_decomp_multi': self.series_decomp_multi.state_dict(),
                      'lstm': self.lstm.state_dict(), 
                      'cnn': self.cnn.state_dict(), 
                      'dense': self.dense.state_dict(), 
                      'dense_softmax': self.dense_softmax.state_dict()}
        torch.save(parameters, filename)
                        
    def forward(self, x): # [nBatch, segLeng, input_dim: 8]
        x = x.float() # 8 features: 일시 (시간 정보), 기온(°C), 1분 강수량(mm), 풍향(deg), 풍속(m/s), 현지기압(hPa), 해면기압(hPa), 습도(%)
        
        if self.in_moving_mean:
            moving_mean, res = self.series_decomp_multi(x)
            x = moving_mean    
        
        if self.feature_wise_norm:
            # Feature-wise Normalization
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            x = (x - x_min) / (x_max - x_min + 1e-7)
            
        x = self.lstm(x) # [nBatch, segLeng, nHidden]
        x = x.permute(0, 2, 1) # [nBatch, nHidden, segLeng]        
        
        x = self.cnn(x) # [nBatch, nb_filters[-1], segLeng]
        x = x.permute(0, 2, 1) # [nBatch,, segLeng, nb_filters[-1]]
        x = self.dropout(x)
        output = self.dense(x) # [nBatch, seqLeng, output_dim]
        
        # attention 
        sof = self.dense_softmax(x)  
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        pred = (output * sof).sum(1) / sof.sum(1)   # [bs, output_dim]
        pred[0:5] = 0 # 0~4: zero padding
        pred[21:] = 0 # 21~23: zero padding
        return pred
        
class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, rec_dropout=0, num_layers=1, in_moving_mean=True, decomp_kernel=[3, 5, 7, 9], feature_wise_norm=True):
        super(RNN, self).__init__()

        self.feature_wise_norm = feature_wise_norm
        self.in_moving_mean = in_moving_mean
        self.decomp_kernel = decomp_kernel
        self.series_decomp_multi = series_decomp_multi(kernel_size=self.decomp_kernel)

        self.rnn = RNNModule(input_dim=input_dim, hidden_dim=hidden_dim, rec_dropout=rec_dropout, num_layers=num_layers) 
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dense_softmax = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1) 
        
    def load_state_dict(self, state_dict, strict=True):
        self.series_decomp_multi.load_state_dict(state_dict["series_decomp_multi"])
        self.rnn.load_state_dict(state_dict["rnn"])
        self.dense.load_state_dict(state_dict["dense"])
        self.dense_softmax.load_state_dict(state_dict["dense_softmax"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"series_decomp_multi": self.series_decomp_multi.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense": self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense_softmax": self.dense_softmax.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'series_decomp_multi': self.series_decomp_multi.state_dict(),
                      'rnn': self.rnn.state_dict(),
                      'dense': self.dense.state_dict(),
                      'dense_softmax': self.dense_softmax.state_dict()}
        torch.save(parameters, filename)
                        
    def forward(self, x): # [nBatch, segLeng, input_dim: 8]
        x = x.float() # 8 features: 일시 (시간 정보), 기온(°C), 1분 강수량(mm), 풍향(deg), 풍속(m/s), 현지기압(hPa), 해면기압(hPa), 습도(%)
        
        if self.in_moving_mean:
            moving_mean, res = self.series_decomp_multi(x)
            x = moving_mean    
        
        if self.feature_wise_norm:
            # Feature-wise Normalization
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            x = (x - x_min) / (x_max - x_min + 1e-7)
            
        x = self.rnn(x) # [nBatch, segLeng, nHidden]

        output = self.dense(x) # [nBatch, seqLeng, output_dim]
        
        # attention 
        sof = self.dense_softmax(x)  
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        pred = (output * sof).sum(1) / sof.sum(1)   # [bs, output_dim]
        pred[0:5] = 0 # 0~4: zero padding
        pred[21:] = 0 # 21~23: zero padding
        return pred
        
class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, rec_dropout=0, num_layers=1, in_moving_mean=True, decomp_kernel=[3, 5, 7, 9], feature_wise_norm=True):
        super(LSTM, self).__init__()

        self.feature_wise_norm = feature_wise_norm
        self.in_moving_mean = in_moving_mean
        self.decomp_kernel = decomp_kernel
        self.series_decomp_multi = series_decomp_multi(kernel_size=self.decomp_kernel)

        self.lstm = LSTMModule(input_dim=input_dim, hidden_dim=hidden_dim, rec_dropout=rec_dropout, num_layers=num_layers) 
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dense_softmax = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1) 
        
    def load_state_dict(self, state_dict, strict=True):
        self.series_decomp_multi.load_state_dict(state_dict["series_decomp_multi"])
        self.lstm.load_state_dict(state_dict["lstm"])
        self.dense.load_state_dict(state_dict["dense"])
        self.dense_softmax.load_state_dict(state_dict["dense_softmax"])
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"series_decomp_multi": self.series_decomp_multi.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "lstm": self.lstm.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense": self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "dense_softmax": self.dense_softmax.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'series_decomp_multi': self.series_decomp_multi.state_dict(),
                      'lstm': self.lstm.state_dict(),
                      'dense': self.dense.state_dict(),
                      'dense_softmax': self.dense_softmax.state_dict()}
        torch.save(parameters, filename)
                        
    def forward(self, x): # [nBatch, segLeng, input_dim: 8]
        x = x.float() # 8 features: 일시 (시간 정보), 기온(°C), 1분 강수량(mm), 풍향(deg), 풍속(m/s), 현지기압(hPa), 해면기압(hPa), 습도(%)
        
        # if self.in_moving_mean:
        #     moving_mean, res = self.series_decomp_multi(x)
        #     x = moving_mean    
        
        # if self.feature_wise_norm:
        #     # Feature-wise Normalization
        #     x_min = x.min(dim=1, keepdim=True)[0]
        #     x_max = x.max(dim=1, keepdim=True)[0]
        #     x = (x - x_min) / (x_max - x_min + 1e-7)
            
        x = self.lstm(x) # [nBatch, segLeng, nHidden]

        y = self.dense(x) # [nBatch, seqLeng, output_dim]
        # pred = y.mean(dim=1) # [nBatch, output_dim]
        
        # attention 
        sof = self.dense_softmax(x)  
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        pred = (y * sof).sum(1) / sof.sum(1)   # [bs, output_dim]
        
        pred[0:5] = 0 # 0~4: zero padding
        pred[21:] = 0 # 21~23: zero padding
        return pred
