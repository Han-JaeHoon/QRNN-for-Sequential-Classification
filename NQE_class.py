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

class my_utils:

    def __init__(self, n_qu):
        self.dev = qml.device('default.qubit', wires = n_qu)            

    def embedding(self, params, n_qu):
        '''
        embedding layer
        '''
        n = n_qu
        for i in range(n):
            qml.Hadamard(i)
            qml.RZ(2.0 * params[ : , i], i)
        
        for i in range(n - 1):
            qml.IsingZZ(2.0 * params[ : , n + i] , [i, i + 1])

    def fidelity(self, vec1, vec2, n_qu):
        @qml.qnode(self.dev, interface = "torch")
        def inner_fidelity(vec1, vec2, n_qu):
            '''
                Args:
                    vec1 : list, (2n - 1)개의 element로 이루어진 vector
                    vec2 : list, (2n - 1)개의 element로 이루어진 vector
            '''
            self.embedding(vec1, n_qu) # Phi(x1) circuit 적용
            qml.adjoint(self.embedding)(vec2, n_qu) # Phi^t(x2) 적용
            return qml.probs()
        return inner_fidelity(vec1, vec2, n_qu)
    def quantum_layer_Z(self, mapped_data1, mapped_data2, parameter1, parameter2, n_qu):
        @qml.qnode(self.dev, interface='torch')
        def inner_quantum_layer_Z(mapped_data1, mapped_data2, parameter1, parameter2, n_qu):
            self.embedding(params=mapped_data1, n_qu=n_qu) #, is_first=True)
            qml.Barrier()
            self.ansatz(params=parameter1, n_qu=n_qu)
            qml.Barrier()
            self.embedding(params=mapped_data2, n_qu=n_qu) #, is_first=True)
            qml.Barrier()
            self.ansatz(params=parameter2, n_qu=n_qu)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qu)]
        return inner_quantum_layer_Z(mapped_data1, mapped_data2, parameter1, parameter2, n_qu)

    def quantum_layer_prob(self, mapped_data1, mapped_data2, parameter1, parameter2, n_qu):
        @qml.qnode(self.dev, interface='torch')
        def inner_quantum_layer_prob(mapped_data1, mapped_data2, parameter1, parameter2, n_qu):
            self.embedding(params=mapped_data1, n_qu=n_qu) #, is_first=True)
            qml.Barrier()
            self.ansatz(params=parameter1, n_qu=n_qu)
            qml.Barrier()
            self.embedding(params=mapped_data2, n_qu=n_qu) #, is_first=True)
            qml.Barrier()
            self.ansatz(params=parameter2, n_qu=n_qu)
            return [qml.expval((qml.PauliZ(i) + qml.Identity(i)) / 2) for i in range(n_qu)]
        return inner_quantum_layer_prob(mapped_data1, mapped_data2, parameter1, parameter2, n_qu)

    def quantum_circuit(self, inputs1, inputs2, weights1, weights2):
        @qml.qnode(self.dev, interface="torch")
        def inner_quantum_circuit(inputs1, inputs2, weights1, weights2):
            block = RNN_block(self.n_qu)
            block.embedding(inputs1)
            block.ansatz(weights1)
            block.embedding(inputs2)
            block.ansatz(weights2)
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        return inner_quantum_circuit(inputs1, inputs2, weights1, weights2)

    def quantum_layer(self, inputs1, inputs2, weights1, weights2):
        return self.quantum_circuit(inputs1, inputs2, weights1, weights2)

    def generate_tensor(self, seed, size):
        """
        주어진 시드와 크기에 맞게 torch.Tensor를 생성합니다.

        Args:
            seed (int): 시드 값
            size (tuple): 생성할 텐서의 크기

        Returns:
            torch.Tensor: 생성된 텐서
        """
        torch.manual_seed(seed)
        return torch.randn(size)

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