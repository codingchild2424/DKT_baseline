import argparse

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

#다른 .py파일 가져오기
from model import DKT
from trainer import Trainer
from dataloader import ASSISTments_data_loader
from utils import collate

#데이터 경로
DATA_DIR = 'data/2015_100_skill_builders_main_problems.csv'

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.6)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--test_ratio', type=float, default=.2)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config

def main(config):

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    #객체 선언
    data_loader = ASSISTments_data_loader(DATA_DIR, device)

    #one_hot_vectors로 만들기
    one_hot_vectors = data_loader.make_one_hot_vectors()
    #batches로 만들기
    batches = data_loader.make_batches(one_hot_vectors)
    #batche 데이터 섞기
    batches = random.sample(batches, len(batches))

    #train, valid, test 비율 나누기
    train_cnt = int(len(batches) * config.train_ratio)
    valid_cnt = int(len(batches) * config.valid_ratio)
    test_cnt = int(len(batches) * config.test_ratio)

    cnts = [train_cnt, valid_cnt, test_cnt]

    #데이터를 나눔
    train_data = batches[:cnts[0]]
    valid_data = batches[cnts[0]:cnts[0] + cnts[1]]
    test_data = batches[cnts[0] + cnts[1]:]

    #hyperparameters
    input_size = len(batches[0][0])
    hidden_size = 50

    #model 선언
    model = DKT(input_size = input_size, hidden_size = hidden_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    #crit을 통해 data_loader.py에 있는 loss_function()을 받아옴
    crit = data_loader.loss_function

    #device를 하나 더 받도록 만듬
    trainer = Trainer(model, optimizer, crit, device)

    trainer.train(train_data, valid_data, config)

    #Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, config.model_fn)

#실행
if __name__ == '__main__':
    config = define_argparser()
    main(config)