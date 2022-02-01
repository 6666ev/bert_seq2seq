import random
import pandas as pd

RANDOM_SEED = 22
random.seed(RANDOM_SEED)

fact, zm, xq = [], [], []

with open("data/laic2021/total.src") as f:
    for line in f.readlines():
        fact.append(line)

with open("data/laic2021/total_zm.tgt") as f:
    for line in f.readlines():
        zm.append(line)

with open("data/laic2021/total_xq.tgt") as f:
    for line in f.readlines():
        xq.append(line)


shuffle_idx = list(range(len(fact)))
random.shuffle(shuffle_idx)


def data_split(idx, data, data_name):
    total_num = len(data)
    train_num = int(total_num * 0.8)
    valid_num = int(total_num * 0.1)
    test_num = total_num - train_num - valid_num
    data = pd.Series(data)  # 67651
    data = data[idx]
    train_data = data[:train_num].tolist()
    valid_data = data[train_num:train_num+valid_num].tolist()
    test_data = data[train_num+valid_num:].tolist()

    path = "data/laic2021/train/{}".format(data_name)
    with open(path, "w") as f:
        for line in train_data:
            line = line.replace("\n", "")
            f.write(line+"\n")

    path = "data/laic2021/valid/{}".format(data_name)
    with open(path, "w") as f:
        for line in valid_data:
            line = line.replace("\n", "")
            f.write(line+"\n")

    path = "data/laic2021/test/{}".format(data_name)
    with open(path, "w") as f:
        for line in test_data:
            line = line.replace("\n", "")
            f.write(line+"\n")


fact = data_split(shuffle_idx, fact, "fact.src")
zm = data_split(shuffle_idx, zm, "zm.tgt")
xq = data_split(shuffle_idx, xq, "xq.tgt")
