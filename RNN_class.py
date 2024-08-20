# Quantum
import pennylane as qml
# numpy
import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Layer
from kan import KAN
from RNN_block import RNN_block
# Data processing
from fucntions import data_seq, train_seq
from utils import my_utils

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0),self.hidden_size)  # 초기 은닉 상태
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 마지막 시퀀스의 출력을 사용
        return out
    
class RNN_layer(nn.Module):

    def __init__(self, input_size, output_size, num_layers, nQE_model = None):
        """_RNN layer

        Args:
            input_size (_int_): _input feature의 개수_
            output_size (_int_): _output feature의 개수_
            num_layers (_int_): _필요한 RNN layer 수_
        """
        super(RNN_layer, self).__init__()
        self.utils = my_utils(input_size)
        self.linear = KAN([input_size, input_size * 2 - 1], grid = 1)
        self.input_size = input_size
        self.output_size = output_size
        self.num_layer = num_layers
        self.cls_layer = nn.Sequential(nn.Linear(input_size,16),nn.ReLU(),nn.Linear(16,1))
        ## QNE 수행할 Linear layer
        self.nqe_model = nQE_model

        ## Ansatz parameter
        self.ansatz_params_1 = nn.Parameter(torch.rand([24],dtype = torch.float32),requires_grad=True)
        self.ansatz_params_2 = nn.Parameter(torch.rand([24],dtype = torch.float32),requires_grad=True)
        self.rnn_layer = self.utils.quantum_layer

        self.chk = True # Forward 실행 시 shape 출력 여부


    def nQE_layer(self, input):
        if self.nqe_model == None:
            n_qu = input.shape[1]
            n_batch = input.shape[0]
            for i in range(n_qu - 1):
                input = torch.cat(([input,((torch.pi-input[:,i]) * (torch.pi-input[:,i+1])).reshape(n_batch,1)]),1)
            return input
        
        if self.nqe_model.mode == 'KAN':
            result = self.nqe_model.linear1(input)
            
        
        if self.nqe_model.mode == 'FC':
            result = self.nqe_model(input)
        return result
    

    def forward(self, inputs, return_hidden_list = False, chk=True):
        """_summary_

        Args:
            inputs (_torch tensor_): _(batch,seq_len,feature_size)_
        """
        if chk and self.chk:
            self.chk = False
        
        batch = inputs.shape[0]
        seq_len = inputs.shape[1]
        initial_t = self.utils.generate_tensor(30, [inputs.shape[0], inputs.shape[2] * 2 - 1]).float()
        inputs = inputs.permute(1, 0, 2)
        ## inputs  = (seq_len,batch,feature_size)
        input = self.nQE_layer(inputs[0])
        
        hidden = torch.stack(self.rnn_layer(initial_t, input, self.ansatz_params_1,self.ansatz_params_2),dim=1).float()
        hidden = hidden.to(torch.float32)
        if return_hidden_list:
            hidden_list = hidden
        for input in inputs[1:]:
            input = self.nQE_layer(input)
            hidden = self.linear(hidden)
            hidden = torch.stack(self.rnn_layer(hidden,input,self.ansatz_params_1,self.ansatz_params_2),dim=1).float()
            hidden = hidden.to(torch.float32)
            if return_hidden_list:
                hidden_list = torch.concat([hidden_list,hidden])
        if return_hidden_list:
            hidden_list = torch.reshape(hidden_list,[batch,seq_len,-1])
            return hidden_list
        return self.cls_layer(hidden)