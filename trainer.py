from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    # train
    def _train(self, x, config):
        self.model.train()

        # 전체 데이터 shuffle하기
        



    def _validate(self, x, config):
        self.model.eval()
        # validate



    def train(self):
        pass



