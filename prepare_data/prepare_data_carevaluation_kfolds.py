import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

def read_file(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        lines = [l for l in lines if l != '']
    return lines

path = '../data/car.data'
data = read_file(path)
data = [l.split(',') for l in data]

# transform categorical features into numeric values
trans = [{'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}, # buying
         {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}, # maint
         {'2': 2, '3': 3, '4': 4, '5more': 5}, # doors
         {'2': 2, '4': 4, 'more': 5}, # persons
         {'small': 1, 'med': 2, 'big': 3}, # lug_boot
         {'low': 1, 'med': 2, 'high': 3} # safety
        ]
features = np.array([[trans[i][v] for i, v in enumerate(l[:-1])] for l in data], dtype='float')

# split data into privileged features and normal features
# according to the estimation time.
# maintenance is privileged feature.
ind_pf = [5]
ind_nf = [i for i in range(6) if i not in ind_pf]

# we work on binary classification for now
# acceptable (1) or not (0).
trans_label = {'unacc': 0, 'acc': 1, 'good': 1, 'vgood': 1}
labels = np.array([trans_label[l[-1]] for l in data], dtype='int')
print(f'labels shape: {labels.shape}')
uniques, counts = np.unique(labels, return_counts=True)
percentages = dict(zip(uniques, counts * 100 / len(labels)))
print(f'Percentages of each value in labels: {percentages}')

# split into 10 folds: 1 for test, 1 for valid, 8 for train
X = features
y = labels

num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
print(f'Split into {skf.get_n_splits(X, y)} folds.')
folds = [test_index for  train_index, test_index in skf.split(X, y)]

for i in range(num_folds):
    test_index = folds[i]
    valid_index = folds[(i + 1) % num_folds]
    train_index = np.concatenate([folds[j] for j in range(num_folds) \
                                           if j not in [i, (i + 1) % num_folds]])
    Path(f'../data/carevaluation/fold{i}').mkdir(parents=True, exist_ok=True)
    # privileged and normal features
    dump_svmlight_file(X[train_index][:, ind_pf], y[train_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.pf.train')
    dump_svmlight_file(X[train_index][:, ind_nf], y[train_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.nf.train')

    dump_svmlight_file(X[valid_index][:, ind_pf], y[valid_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.pf.valid')
    dump_svmlight_file(X[valid_index][:, ind_nf], y[valid_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.nf.valid')

    dump_svmlight_file(X[test_index][:, ind_pf], y[test_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.pf.test')
    dump_svmlight_file(X[test_index][:, ind_nf], y[test_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.nf.test')
    # all features
    dump_svmlight_file(X[train_index], y[train_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.train')
    dump_svmlight_file(X[valid_index], y[valid_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.valid')
    dump_svmlight_file(X[test_index], y[test_index], \
                       f'../data/carevaluation/fold{i}/carevaluation.txt.test')

