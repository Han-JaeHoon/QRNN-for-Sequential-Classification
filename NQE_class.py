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
    def __init__(self, n_feature, mode : str):
        '''
            Args:
                type(str) : 'FC' or 'KAN'
                n_feature(int) : # of feature
        '''
        super(NQE, self).__init__()

        self.mode = mode
        self.utils = my_utils(n_qu = n_feature)

        if mode == 'FC':
            self.n_qu = n_feature
            self.li1 = nn.Linear(n_feature, n_feature * n_feature)
            self.li2 = nn.Linear(n_feature * n_feature, n_feature * n_feature)
            self.li3 = nn.Linear(n_feature * n_feature, 2 * n_feature - 1)
            self.quantum_layer = self.utils.fidelity
        
        if mode == 'KAN':
            self.n_qu = n_feature
            self.linear1 = KAN([self.n_qu, self.n_qu * 2 + 1, self.n_qu * 2 - 1], grid = 1)
            self.quantum_layer = self.utils.fidelity

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

    def forward(self, inputs):
        if self.mode == 'FC':
            return self.forward_FC(inputs)
        if self.mode == 'KAN':
            return self.forward_KAN(inputs)