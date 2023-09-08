import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

def read_file(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        lines = [l for l in lines if l != '']
    return lines

path = '../data/spambase.data'
data = read_file(path)

data = [l.split(',') for l in data]

features = np.array([l[:-1] for l in data], dtype='float')

# split data into previleged features and normal features
# according to their correlation (importance) >= 0.2.
ind_pf = [4, 5, 6, 7, 8, 10, 15, 16, 17, 18, 20, 22, 23, 24, 25, 51, 52, 55, 56]
ind_nf = [i for i in range(57) if i not in set(ind_pf)]

# we work on binary classification for now
# spam (1) or not (0).
labels = np.array([l[-1] for l in data], dtype='int')
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
    Path(f'../data/spambase/fold{i}').mkdir(parents=True, exist_ok=True)
    # privileged and normal features
    dump_svmlight_file(X[train_index][:, ind_pf], y[train_index], \
                       f'../data/spambase/fold{i}/spambase.txt.pf.train')
    dump_svmlight_file(X[train_index][:, ind_nf], y[train_index], \
                       f'../data/spambase/fold{i}/spambase.txt.nf.train')

    dump_svmlight_file(X[valid_index][:, ind_pf], y[valid_index], \
                       f'../data/spambase/fold{i}/spambase.txt.pf.valid')
    dump_svmlight_file(X[valid_index][:, ind_nf], y[valid_index], \
                       f'../data/spambase/fold{i}/spambase.txt.nf.valid')

    dump_svmlight_file(X[test_index][:, ind_pf], y[test_index], \
                       f'../data/spambase/fold{i}/spambase.txt.pf.test')
    dump_svmlight_file(X[test_index][:, ind_nf], y[test_index], \
                       f'../data/spambase/fold{i}/spambase.txt.nf.test')
    # all features
    dump_svmlight_file(X[train_index], y[train_index], \
                       f'../data/spambase/fold{i}/spambase.txt.train')
    dump_svmlight_file(X[valid_index], y[valid_index], \
                       f'../data/spambase/fold{i}/spambase.txt.valid')
    dump_svmlight_file(X[test_index], y[test_index], \
                       f'../data/spambase/fold{i}/spambase.txt.test')

