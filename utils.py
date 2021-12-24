import numpy as np
import pandas as pd
import torch

# **데이터 변환**

# 2015 데이터를 받기
# 데이터 경로는 확인 필요
data_2015_path = 'data/2015_100_skill_builders_main_problems.csv'
# 판다스로 데이터를 불러옴
data_pd = pd.read_csv(data_2015_path)
# 데이터 중에서 user_id, sequence_id, correct 세 가지 열을 가져옴
data_pd = data_pd.loc[:, ['user_id', 'sequence_id', 'correct']]
# 판다스 데이터프레임을 넘파이로 변환
data_np = data_pd.to_numpy()
# 안에 있는 인자들을 int64로 바꿈
data_np = data_np.astype(np.int64)

#data_np에서 각 열을 students, items, answers로 나눠서 담음
students = data_np[:, 0]
items = data_np[:, 1]
answers = data_np[:, 2]

# **변수 정리**

#n_students: 중복되지 않은 학생의 수만 담은 변수
n_students = np.unique(students, return_counts = True)
n_students = n_students[0].shape[0]

#n_items: 중복되지 않은 문항의 수만 담은 변수
n_items = np.unique(items, return_counts=True)
n_items = n_items[0].shape[0]

#idx_students: 중복되지 않은 학생들의 목록
idx_students = np.unique(students)
#idx_items: 중복되지 않은 문항들의 목록
idx_items = np.unique(items)

# **one_hot_vectors 생성**

# one_hot_vectors를 만들기 위해, 문항의 수만큼 0으로 채워져있는 벡터를 만듦
one_hot_vectors = np.zeros((len(items), n_items * 2))

for i, data in enumerate(data_np):
    # 첫번째 행의 문항이 idx_items의 몇 번째 인덱스인지 파악해서 idx에 저장
    # data_np[i][1]은 i행의 문항임
    idx = list(idx_items).index(data_np[i][1])
    # 정답이라면 one_hot_vectors에 처음~M(문항의 갯수)에 1을 더하고,
    one_hot_vectors[i, idx] += data_np[i][2]
    # 오답이면 one_hot_vectors의 M+1~2M에 1을 더함
    one_hot_vectors[i, idx + n_items] += 1 - data_np[i][2]

# **batch 만들기**

# 리스트로 변경하기
students = list(students)
one_hot_vectors = list(one_hot_vectors)
idx_students = list(idx_students)

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


#최종적으로 함수를 만들어서 batches를 반환하도록 만들기