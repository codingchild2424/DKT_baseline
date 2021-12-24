import argparse

import random

import torch
import torch.nn as nn
import torch.optim as optim

#다른 .py파일 가져오기
from model import DKT
from trainer import Trainer
from dataloader import ASSISTments_data_loader

#데이터 경로
DATA_DIR = 'data/2015_100_skill_builders_main_problems.csv'

#객체 선언
data_loader = ASSISTments_data_loader(DATA_DIR)

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

    #데이터를 나누면서 device에 올림
    train_data = batches[:cnts[0]].to(device)
    valid_data = batches[cnts[0]:cnts[0] + cnts[1]].to(device)
    test_data = batches[cnts[0] + cnts[1]:].to(device)

    model = DKT()











    #여기서 model과 data를 device에 올리기!!!





    #train, valid, test 세트로 나누기


if __name__ == '__main__':
    config = define_argparser()
    main(config)