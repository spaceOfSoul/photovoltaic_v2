import torch
from torch import nn as nn

class RNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0, num_layers=1):
        super(RNNModule, self).__init__()

        # 기본 RNN 레이어 추가
        self.rnn = nn.RNN(input_dim, hidden_dim, bidirectional=False, dropout=rec_dropout, batch_first=True, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent

class LSTMModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, rec_dropout=0, num_layers=1):
        super(LSTMModule, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=False, batch_first=True,
                           dropout=rec_dropout, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent
