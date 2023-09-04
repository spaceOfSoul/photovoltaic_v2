import torch.nn as nn
import torch

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 1))
        lin = lin.permute(0, 2, 1)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 1))
        lin = lin.permute(0, 2, 1)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

class CNNModule(nn.Module):

    def __init__(self, n_in_channel, activ="Relu", conv_dropout=0,
                 kernel_size=7*[3], padding=7*[1], stride=7*[1], nb_filters=7*[23],
                 pooling=7*[1]
                 ):
        super(CNNModule, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv1d(nIn, nOut, kernel_size[i], stride[i], padding[i]))  # Conv1d here
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm1d(nOut, eps=0.001, momentum=0.99))  # BatchNorm1d here
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(i),
                               nn.Dropout(dropout))

        batch_norm = True
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activ)
            cnn.add_module('pooling{0}'.format(i), nn.AvgPool1d(pooling[i]))  # MaxPool1d here

        self.cnn = cnn

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames)
        # conv features
        # print("cnn_in: ", x.shape)
        x = self.cnn(x)
        return x


