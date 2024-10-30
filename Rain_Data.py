# # Quantum
# import pennylane as qml

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Numpy, Pandas
import numpy as np
import pandas as pd

# # Layer
# from kan import KAN
# from RNN_block import RNN_block

# Data processing
# from fucntions import data_seq, train_seq
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# # Quantum User-Def Classes
# from utils import my_utils
# from NQE_class import NQE
# from NQE_train_class import NQE_Train

class Train_Data:
    '''
        preprocess된 CSV 파일(train, label)을 입력받아, 
        PCA를 통해 feature의 개수를 원하는 만큼 줄이고
        배치된 데이터를 생성하여 바로 학습할 수 있도록 데이터를 준비해주는 class
    '''

    def __init__(self, train_path : str, label_path : str, required_data : int, required_components = 4):
        '''
            train_path(str) : path of train data csv file (ex : ./data/train_data_Darwin.csv)
            label_path(str) : path of label data csv file (ex : ./data/label_data_Darwin.csv)
        '''
        if required_data is None:
            required_data = len(self.train_df)
        
        self.pca = PCA(n_components=required_components)

        self.train_df = pd.read_csv(train_path)[:required_data]
        self.label_df = pd.read_csv(label_path)[:required_data]
        self.train_data = torch.tensor(self.pca.fit_transform(self.train_df)).to(torch.float)
        self.label_data = torch.tensor(self.label_df.to_numpy()).to(torch.float)
