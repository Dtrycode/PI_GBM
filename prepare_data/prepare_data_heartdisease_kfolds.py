import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

def read_file(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        lines = [l for l in lines if l != '']
    return lines

path = '../data/processed.cleveland.data'
data = read_file(path)

# filter out missing value instances
print(f'Number of instances loaded: {len(data)}')
data = [l.split(',') for l in data if '?' not in l]
print(f'Number of instances (no missing): {len(data)}')

features = np.array([l[:-1] for l in data], dtype='float')

# split data into previleged features and normal features
# according to the examination cost.
# first 6 features are normal, the last 7 features are privileged.
ind_pf = range(6, 13)
ind_nf = range(6)

# we work on binary classification for now
# heart-disease (1) or not (0).
labels = np.array([l[-1] for l in data], dtype='int')
labels[labels > 0] = 1
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
    Path(f'../data/heartdisease/fold{i}').mkdir(parents=True, exist_ok=True)
    # privileged and normal features
    dump_svmlight_file(X[train_index][:, ind_pf], y[train_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.pf.train')
    dump_svmlight_file(X[train_index][:, ind_nf], y[train_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.nf.train')

    dump_svmlight_file(X[valid_index][:, ind_pf], y[valid_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.pf.valid')
    dump_svmlight_file(X[valid_index][:, ind_nf], y[valid_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.nf.valid')

    dump_svmlight_file(X[test_index][:, ind_pf], y[test_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.pf.test')
    dump_svmlight_file(X[test_index][:, ind_nf], y[test_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.nf.test')
    # all features
    dump_svmlight_file(X[train_index], y[train_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.train')
    dump_svmlight_file(X[valid_index], y[valid_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.valid')
    dump_svmlight_file(X[test_index], y[test_index], \
                       f'../data/heartdisease/fold{i}/heartdisease.txt.test')

