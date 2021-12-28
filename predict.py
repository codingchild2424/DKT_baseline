import torch
import torch.nn
from copy import deepcopy

import numpy as np
from sklearn import metrics

from tqdm import tqdm

from torch.utils.data import DataLoader

#utils에서 collate를 가져옴
from utils import DKT_utils
from utils import collate

#데이터 경로
DATA_DIR = 'data/2015_100_skill_builders_main_problems.csv'

class Predictor:

    def __init__(self, model, optimizer, crit, device):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.device = device

        super().__init__()

    def _predict(self, test_data, config):
        self.model.eval()

        # validate 데이터 shuffle하기
        test_loader = DataLoader(
            dataset = test_data,
            batch_size = config.batch_size, #batch_size는 config에서 받아옴
            #test_data이기에 shuffle False로 잡음
            shuffle = False,
            collate_fn = collate
        )

        auc_score = 0

        #y_true and score 때문에 가져왔는데, 나중에 함수 나눠서 없애고 정리하기
        dkt_utils = DKT_utils(DATA_DIR, self.device)

        y_trues, y_scores = [], []

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(self.device)
                y_hat_i = self.model(data)
                loss = self.crit(y_hat_i[:-1], data[1:])
                #y_true값과 y_score값을 계산
                y_true, y_score = dkt_utils.y_true_and_score(data, y_hat_i)

                y_trues.append(y_true)
                y_scores.append(y_score)
            
        #y_true와 y_score를 numpy로 바꿈
        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        #train의 auc_score를 계산
        auc_score += metrics.roc_auc_score(y_trues, y_scores)

        return auc_score

    def predict(self, test_data, config):

        print("========================Predict Start===========================")

        highest_auc_score = 0

        for epoch_index in tqdm(range(config.n_epochs), ascii = True):
            test_auc_score = self._predict(test_data, config)

            if test_auc_score >= highest_auc_score:
                highest_auc_score = test_auc_score

        print("========================Predict Finish===========================")
        print("\n")
        print("The Highest_Auc_Score in Predicting Session is %.4e" % (
                highest_auc_score,
            ))
        print("\n")
    