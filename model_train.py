import pickle
import numpy as np
from sklearn.model_selection import ShuffleSplit
from model import CrnnCrispr
def onehot(x):
    z = list()
    for y in list(x):
        if y in "0" : z.append(0)
        elif y in "1" : z.append(1)
        elif y in "2" : z.append(2)
        elif y in "3": z.append(3)
        else:
            print("Non-0123")      
    return z

def set_data(x1, s):
    for num, pos in enumerate(onehot(s)):
        x1[num][pos] = 1

def load_data_kf(x1, x2, y):
    train_test_data = []
    kf = ShuffleSplit(n_splits=5, test_size=0.15, random_state=33)
    for train_index, test_index in kf.split(X):
        x1_train, x1_test = x1[train_index], x1[test_index]
        x2_train, x2_test = x2[train_index], x2[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_test_data.append((x1_train, x1_test, x2_train, x2_test, y_train, y_test))
    return train_test_data

# one-hot encoding
f = open('data_onehot.csv')
lines = f.readlines()
f.close()
l = len(lines)
x1 = np.zeros((l, 23, 4), dtype=int)
for num, line in enumerate(lines):
    line = line[:-1]
    set_data(x1[num], line)

# label encoding
with open('data_label.pkl', 'rb') as f1:
    label_data = pickle.load(f1)
x2,  y = label_data[0], label_data[1]

data_list = load_data_kf(x1, x2, y)
param = {'model_type':'data',
     'batch_size': 500,
     'epochs': 200}
for data in data_list:
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = data[0], data[1], data[2], data[3], data[4], data[5]
    model = CrnnCrispr(x1_train, x2_train, y_train, n, **param)
    n = n+1
