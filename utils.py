import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def collate(batch):
    #가변길이 함수에 padding을 넣어주는 것
    #https://runebook.dev/ko/docs/pytorch/generated/torch.nn.utils.rnn.pad_sequence
    return nn.utils.rnn.pad_sequence(batch)

# ASSISTments2015_data_loader class 정의
class DKT_utils:

    def __init__(self, DATA_DIR, device):
        self.data_path = DATA_DIR
        #해당 부분을 간결하게 만들 수 있도록 정의해보기
        self.data_pd = pd.read_csv(self.data_path)
        self.data_np = self.data_pd.loc[:, ['user_id', 'sequence_id', 'correct']].to_numpy().astype(np.int64)
        #각각의 인스턴스 정의
        self.students = self.data_np[:, 0]
        self.items = self.data_np[:, 1]
        self.answers = self.data_np[:, 2]
        #n_students: 중복되지 않은 학생의 수만 담은 변수
        self.n_students = np.unique(self.students, return_counts = True)[0].shape[0]
        #n_items: 중복되지 않은 문항의 수만 담은 변수
        self.n_items = np.unique(self.items, return_counts=True)[0].shape[0]
        #idx_students: 중복되지 않은 학생들의 목록
        self.idx_students = np.unique(self.students)
        #idx_items: 중복되지 않은 문항들의 목록
        self.idx_items = np.unique(self.items)
        self.device = device

    # one_hot_vectors를 만드는 method
    def make_one_hot_vectors(self):
        one_hot_vectors = np.zeros((len(self.items), self.n_items * 2))

        for i, data in enumerate(self.data_np):
            # 첫번째 행의 문항이 idx_items의 몇 번째 인덱스인지 파악해서 idx에 저장
            # data_np[i][1]은 i행의 문항임
            idx = list(self.idx_items).index(self.data_np[i][1])
            # 정답이라면 one_hot_vectors에 처음~M(문항의 갯수)에 1을 더하고,
            one_hot_vectors[i, idx] += self.data_np[i][2]
            # 오답이면 one_hot_vectors의 M+1~2M에 1을 더함
            one_hot_vectors[i, idx + self.n_items] += 1 - self.data_np[i][2]

        return one_hot_vectors

    # batches를 만드는 method, one_hot_vectors를 사용해야 함
    def make_batches(self, one_hot_vectors):
        students = list(self.students)
        one_hot_vectors = list(one_hot_vectors)
        idx_students = list(self.idx_students)

        students_id_to_idx = {}

        for i, unique_student in enumerate(idx_students):
            students_id_to_idx[unique_student] = i

        # batches 만들기
        # batches는 []안에 unique한 학생의 수만큼의 빈 []를 넣어두는 변수
        # 이후에 각 학생별 [] 안에 학생이 푼 one_hot_vector를 집어넣을 것임
        dummy_batches = []

        # dummy_batches에 학생의 유니크 한 수만큼의 빈 리스트 넣기
        for i in range(len(idx_students)):
            dummy_batches.append([])

        # 학생의 인덱스에 맞는 빈 리스트에 one_hot_vector를 리스트형태로 넣기
        for i in range(len(students)):
            idx = students_id_to_idx[students[i]]
            dummy_batches[idx].append(one_hot_vectors[i])

        # 진짜 batches를 받을 수 있도록 처리
        batches = []

        # dummy_batches에서 학생별로 데이터를 꺼내서 torch.Tensor로 바꿈
        for batch in dummy_batches:
            batches.append(torch.Tensor(batch))

        return batches

    #data와 예측값 y_hat_i
    def y_true_and_score(self, data, y_hat_i):
        #correc와 mask를 정의
        #correct는 처음부터 M까지의 데이터, 즉 정답값에 속하는 원핫벡터(100개)를 의미함
        correct = data[:,:, :self.n_items]
        #mask는 정오답에 상관없이 앞의 M개 데이터(정답)와 뒤의 M개 데이터(오답)를 더해서 문항의 위치를 확인하기 위한 용도임
        mask = (data[:,:, :self.n_items] + data[:,:, self.n_items:]).type(torch.BoolTensor)
        #y_true에 들어가는 것은 정답값임
        #이 중에서 correct[1:]은 가장 첫번째 정답값을 제하고, 두번째 정답부터 끝까지를 담고 있음
        #mask[1:]을 통해 해당 문항이 정답인지 오답인지를 담고 있는 값을 만들 수 있음 -> 값은 정답이면 1, 아니면 0을 담고 있음
        y_true = torch.masked_select(correct[1:], mask[1:])
        #y_hat_i의 값은 각각 한칸 뒤의 문항에 대한 정답률을 추정하는 확률값임
        #따라서 y_hat_i[:-1]를 통해 마지막 값은 무시하면 두번째 문항부터 마지막 문항까지 예측 확률값을 알 수 있음
        #그래서 mask[1:]을 사용하면, 2번 문항부터의 문항번호를 알 수 있기에 해당 문항의 예측 확률값만 얻을 수 있음
        y_score = torch.masked_select(y_hat_i[:-1], mask[1:])

        return y_true, y_score

    def loss_function(self, y_hat, y_real):
        eps = 1e-8
        #delta는 target이 총 길이가 n_items * 2인데, 
        #이 중에서 앞에 있는 절반([:, :, self.n_items])과
        #뒤에 있는 절반([:, :, self.n_items:])을 합치는 것
        #delta는 결국 one-hot encoding이지만, 차원은 n_itmes임
        #delta = [0, 0, 1, 0 ....] 길이는 n_items
        #delta는 정오답 상관없이 문항의 위치를 알려주는 변수
        delta = y_real[:,:, :self.n_items] + y_real[:,:, self.n_items:]
        #sum(axis=-1)은 차원의 마지막을 기준으로 합치는 것
        #즉, 원핫 벡터의 구성요소를 다 더한 것을 bool로 표시
        #tensor([[True, True, True], [True, True, True]])
        mask = delta.sum(axis=-1).type(torch.BoolTensor).to(self.device)
        #data는 모델 통과 후의 값인데, 차원은 n_items이고, sigmoid()를 통과하여 확률값으로 표시됨
        #여기에 엄청나게 작은 값을 곱하고 더함으로써 매우 작은 값으로 보정함
        #!!!이유는 모르겠음
        y_hat =  y_hat* (1-2*eps) + eps
        #correct는 처음~M까지의 데이터로 여기에는 정답일 경우, 문항 번호를 표기하는 곳임
        #차원은 n_items
        correct = y_real[:,:, :self.n_items].to(self.device)
        #correct는 정답에 대한 정오답 벡터(n_items 차원)
        #data.log()는 각 문항에 대한 확률값의 로그값(n_items 차원)
        #공식은 BCE 공식 그대로
        bce = - correct*y_hat.log() - (1-correct)*(1-y_hat).log()
        #bce값을 delta(정답쪽 one-hot과 오답쪽 one-hot을 더한 값, n_items 차원)와 곱한 후
        #이것을 axis -1 방향으로 더함
        #그러면 해당 문항의 bce 값을 알 수 있고, 모든 문항에 대한 확률값을 알 수 있음
        bce = (bce*delta).sum(axis=-1)
        #최종 반환 값에서 bce를 mask를 통해 모두 선택하고, 이를 평균내어 반환함
        return torch.masked_select(bce, mask).mean()

