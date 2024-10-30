# Quantum
import pennylane as qml
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Numpy, Pandas
import numpy as np
import pandas as pd
# Layer
from kan import KAN
from RNN_block import RNN_block
# Data processing
from fucntions import data_seq, train_seq
from utils import my_utils

class NQE(nn.Module):
    def __init__(self, n_feature, mode : str, rnn_sequence = 0):
        '''
            Args:
                type(str) : 'FC' or 'KAN'
                n_feature(int) : # of feature
                rnn_args(int) : # of sequence if the mode is RNN
        '''
        super(NQE, self).__init__()

        self.mode = mode
        self.utils = my_utils(n_qu = n_feature)
        self.n_qu = n_feature
        self.quantum_layer = self.utils.fidelity

        if mode == 'FC':
            self.li1 = nn.Linear(n_feature, n_feature * n_feature)
            self.li2 = nn.Linear(n_feature * n_feature, n_feature * n_feature)
            self.li3 = nn.Linear(n_feature * n_feature, 2 * n_feature - 1)
        
        if mode == 'KAN':
            self.linear1 = KAN([self.n_qu, self.n_qu * 2 + 1, self.n_qu * 2 - 1], grid = 1)
        
        if mode == 'RNN':
            self.linear1 = nn.RNN(input_size=n_feature, hidden_size=(2 * n_feature - 1), num_layers=rnn_sequence, batch_first=True)

    def forward_input_FC(self, inputs):
        inputs = self.li1(inputs)
        inputs = F.relu(inputs)
        inputs = self.li2(inputs)
        inputs = F.relu(inputs)
        inputs = self.li3(inputs)
        result = 2 * torch.pi * F.relu(inputs)
        return result

    def forward_FC(self,inputs):
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = self.forward_input_FC(input1)
        input2 = self.forward_input_FC(input2)
        output = self.quantum_layer(input1, input2, self.n_qu)[ : , 0]
        return output

    def forward_KAN(self, inputs):
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = self.linear1(input1)
        input2 = self.linear1(input2)
        output = self.quantum_layer(input1, input2, self.n_qu)[ : , 0]
        return output

    def forward_input_RNN(self, inputs):
        output, _ = self.linear1(inputs) # _ is hidden
        output = output.transpose(0, 1)
        result = 2 * torch.pi * F.relu(output[-1])
        return result
    
    def forward_RNN(self, inputs):
        # print('inputs shape :', inputs.shape)
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = self.forward_input_RNN(input1)
        input2 = self.forward_input_RNN(input2)
        # print('input vec1 shape :', input1.shape)
        output = self.quantum_layer(input1, input2, self.n_qu)[ : , 0]
        return output

    def forward(self, inputs):
        if self.mode == 'FC':
            return self.forward_FC(inputs)
        if self.mode == 'KAN':
            return self.forward_KAN(inputs)
        if self.mode == 'RNN':
            return self.forward_RNN(inputs)