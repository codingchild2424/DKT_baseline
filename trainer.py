from copy import deepcopy

import numpy as np
from sklearn import metrics

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

#utils에서 collate를 가져옴
from utils import collate
from utils import loss_function
from utils import y_true_and_score

from dataloader import ASSISTments_data_loader

dataloader = ASSISTments_data_loader

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    #_train
    def _train(self, train_data, config):
        self.model.train()
        # train 데이터 shuffle하기
        train_loader = DataLoader(
            dataset = train_data,
            batch_size = config.batch_size, #batch_size는 config에서 받아옴
            shuffle = True,
            collate_fn = collate
        )

        auc_score = 0

        # train_loader에서 미니배치가 반환됨
        for i, data in enumerate(train_loader):
            #data와 model은 train에서 device에 올릴 것임
            y_hat_i = self.model(data) #|y_hat_i| = torch.Size([190, 100]), 각 값은 문항별 확률값
            config.optimizer.zero_grad()
            #loss를 구하기 위해서는 반환된 값의 차원(n_items)과 다음 값의 차원(n_items / 2)로 설정해서 비교해야함
            #따라서 mask를 씌우는 작업이 필요함
            #해당 기능은 loss_function에서 구현함
            loss = loss_function(y_hat_i[:-1], data[1:])
            loss.backward()
            config.optimizer.step()
            #y_true값과 y_score값을 계산
            y_true, y_score = dataloader.y_true_and_score(data, y_hat_i)
            #train의 auc_score를 계산
            auc_score += metrics.roc_auc_score(y_true, y_score)

            if config.verbose >= 2:
                print("train iteration(%d/%d): auc score=%.4e" % (i + 1, len(train_data), float(auc_score)))

        return auc_score

    #_validate
    def _validate(self, valid_data, config):
        self.model.eval()

        # validate 데이터 shuffle하기
        valid_loader = DataLoader(
            dataset = valid_data,
            batch_size = config.batch_size, #batch_size는 config에서 받아옴
            shuffle = True,
            collate_fn = collate
        )

        auc_score = 0

        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                y_hat_i = self.model(data)
                loss = loss_function(y_hat_i[:-1], data[1:])
                #y_true값과 y_score값을 계산
                y_true, y_score = dataloader.y_true_and_score(data, y_hat_i)
                #train의 auc_score를 계산
                auc_score += metrics.roc_auc_score(y_true, y_score)

                if config.verbose >= 2:
                    print("valid iteration(%d/%d): auc score=%.4e" % (i + 1, len(valid_data), float(auc_score)))

        return auc_score

    # _train과 _validate를 활용해서 train함
    def train(self, train_data, valid_data, config):

        highest_auc = 0
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_auc_score = self._train(train_data, config)
            valid_auc_score = self._valid(valid_data, config)

            if valid_auc_score >= highest_auc:
                highest_auc_score = valid_auc_score
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_auc_score=%.4e  valid_auc_score=%.4e  highest_auc_score=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_auc_score,
                valid_auc_score,
                highest_auc_score,
            ))

        # 가장 최고의 모델 복구    
        self.model.load_state_dict(best_model)

        


