from kan import KAN
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
    
class data_seq():
    def __init__(self, feature_data, label_data):
        self.feature = feature_data
        self.label = label_data
    
    def split_data(self, test_ratio = 0.2, batch_size = 32, seq_first = False, seed = 40):
        """
        데이터를 훈련 및 테스트 세트로 나누고 배치 단위로 나누어 줍니다.

        Args:
            features (torch.Tensor): 입력 특징 텐서
            labels (torch.Tensor): 레이블 텐서
            test_ratio (float): 테스트 세트 비율 (기본값: 0.2)
            batch_size (int): 배치 크기 (기본값: 32)

        Returns:
            train_loader (DataLoader): 훈련 데이터 로더
            test_loader (DataLoader): 테스트 데이터 로더
        """
        if seq_first:
            self.feature = self.feature.permute(1, 0, 2)
            self.label = self.label.permute(1, 0)
        
        # 데이터셋 생성
        dataset = TensorDataset(self.feature, self.label)

        # 훈련 및 테스트 데이터셋 크기 계산
        test_size = int(len(dataset) * test_ratio)
        train_size = len(dataset) - test_size

        # 데이터셋 분할
        torch.manual_seed(seed)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

class train_seq():  # NQE_train
    def __init__(self, model, train_loader, test_loader):
        """data seq로 만든 데이터를 입력하면 훈련을 진행해줍니다.

        Args:
            model (_model_): _description_
            train_loader (_type_): _description_
            test_loader (_type_): _description_
        """
        # super.__init__(**kwargs)
        self.model= model
        self.train_data = train_loader
        self.test_data = test_loader
    
    def train(self, epochs, optimizer, criterion,metrics = [], seq_first = False):
        self.train_loss_list = []
        self.test_loss_list = []
        for i in range(len(metrics)):
          setattr(self,f'metric{i}_list',[])
          setattr(self,f'metric{i}_test_list',[])
        for epoch in range(epochs):
            pred_list = []
            label_list = []
            setattr(self,f'metric{i}',[])
            for train,label in self.train_data:
                config = {}
                if seq_first:
                    train = train.permute(1, 0, 2)
                    label = label.permute(0, 1)
                    
                optimizer.zero_grad()
                pred = self.model(train) ## Forward 사용
                pred_list.append(pred)
                label_list.append(label)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                config['loss'] = loss
                for index,metric in enumerate(metrics):
                  metric_result = metric(pred,label)
                  getattr(self,f'metric{index}').append(metric_result)
                  config[f'metric{index}'] = metric_result
                self.call_message(config)
            config = {}
            print('\n')
            loss_test = self.test(criterion,seq_first)
            self.train_loss_list.append(loss.detach().item())
            self.test_loss_list.append(loss_test.detach().item())
            config['train_loss'] = loss
            config['test_loss'] = loss_test
            for index,metric in enumerate(metrics):
              metric_test = self.test(metric,seq_first)
              metric_loss = getattr(self,f'metric{index}')[-1]
              getattr(self,f'metric{index}_test_list').append(metric_loss)
              config['train_metric'] = metric_loss
              config['test_metric'] = metric_test
            self.call_message(config)
            print('\n')
    def call_message(self,config):
      meassage = '\r'
      for key in config.keys():
        meassage += f' {key} : {config[key]:.5f}'
      sys.stdout.write(meassage)
      

    def test(self, criterion, seq_first = False):
        pred_list = []
        label_list = []
        for test, label in self.test_data:
            if seq_first:
                test = test.permute(1,0,2)
                label = label.permute(0,1)
            pred = self.model(test)
            pred_list.append(pred)
            label_list.append(label)
        pred = torch.concat(pred_list)
        label = torch.concat(label_list)
        loss = criterion(pred, label)
        return loss
    

# class NQE_train()