#trainer에서는 직접적으로 모델을 훈련시키는 코드를 작성

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

    def _train(self, x, config):
        self.model.train()

        #shuffle