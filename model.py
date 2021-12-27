import torch
import torch.nn as nn

# 아래에 모델 클래스 정의
class DKT(nn.Module):
    # 생성자 정의
    def __init__(
        self,
        input_size, # 본 연구에서는 RNN 계열인 LSTM을 사용함, 여기서 받을 input_size의 dim은 3이어야 함
        hidden_size, # RNN은 구조상 hidden_size와 output_size가 같음, 따라서 output_size는 정의할 필요가 없음
        n_layers = 4, # n_layers는 RNN의 층이 몇 개인지를 정의함
        dropout_p = .2, # dropout_p는 모델이 model.train()일때 활성화됨
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__() # nn.Module을 상속받았기에

        # RNN 층을 선언, 저자들의 code에서도 LSTM을 사용했기에 LSTM을 사용함
        self.rnn = nn.LSTM(
            input_size = input_size, # input_size는 문항의 갯수(M)의 두배인 2M이어야 함
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = False, # batch_first = True를 사용하면, batch 크기가 가장 먼저 나옴 / dataloader에 data가 들어가면 어차피 순서가 바뀌어서 batch_first를 False로 사용함
            dropout = dropout_p
        )
        self.layers = nn.Sequential(
            # <입력크기>
            # rnn 층에서 hidden_size로 결과를 출력하므로,
            # Linear층에서는 hidden_size로 입력을 받음
            # <출력크기>
            # 출력은 각 문항에 대한 예측 정답률이 되어야 하므로, 
            # 문항(M)의 갯수만큼 출력하기 위해 int(input_size / 2)로 정의함
            nn.Linear(hidden_size, int(input_size / 2)),
            # DKT는 기본적으로 이진분류 모델이므로, Sigmoid를 사용함
            # Linear를 통과한 값을 Sigmoid로 감싸서 확률값으로 나오게 함
            nn.Sigmoid()
        )

    # forward method 정의
    def forward(self, x):
        # |x| = (sequence_length, batch_size, hidden_size) = (sl, bs, hs)
        # rnn의 ouput으로는 output 결과물과 함께 (h_n)이 나옴 / (h_n): final hidden state for each element in the batch
        # 뒤는 필요없으므로, _로 무시함
        z, _ = self.rnn(x)
        # |z| = (sequence_length, batch_size, hidden_size)
        # z는 모든 time step의 결과를 가져옴
        y = self.layers(z)
        # |y| = (batch_size, int(input_size / 2))
        return y