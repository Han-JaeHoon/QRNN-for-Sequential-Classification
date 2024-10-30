# PyTorch
import torch.optim as optim
# Data processing
from fucntions import data_seq, train_seq

class NQE_Train:
    def __init__(self, nqe, criterion, train_loader,test_loader, matrics = []):
        '''
            Args:
                nqe (NQE) : nqe object want to train
                criterion (function) : loss function
                data_pretrain (data_seq) : want to make train_seq
                optimizer (torch.optimizer)
        '''
        self.nqe = nqe
        self.loss = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optim = optim.Adam(self.nqe.parameters(), lr = 0.005)
        self.matrics = matrics
        
    def train(self, epoch, seq_first = False):
        nqe_seq = train_seq(self.nqe, self.train_loader, self.test_loader)
        nqe_seq.train(epoch, self.optim, self.loss, self.matrics, seq_first=seq_first)
        return self.nqe