import torch
import torch.nn as nn

def collate(batch):
    #가변길이 함수에 padding을 넣어주는 것
    #https://runebook.dev/ko/docs/pytorch/generated/torch.nn.utils.rnn.pad_sequence
    return nn.utils.rnn.pad_sequence(batch)



