import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

def read_file(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        lines = [l for l in lines if l != '']
    return lines

path = '../data/adult.data'
data = read_file(path)

# filter out missing value instances
print(f'Number of instances loaded: {len(data)}')
data = [[d.strip() for d in l.split(',')] for l in data if '?' not in l]
print(f'Number of instances (no missing): {len(data)}')

# categorical features
# workclass, education, marital-status, occupation, relationship, race, sex, native-country
ind_cat = [1, 3, 5, 6, 7, 8, 9, 13]
# numeric features
# age, education-num, capital-gain, capital-loss, hours-per-week
ind_numer = [0, 4, 10, 11, 12]
# labels
# income
ind_label = [14]

data = np.array(data)
data_cat = data[:, ind_cat]
enc = OrdinalEncoder()
data_cat = enc.fit_transform(data_cat)

data_numer = np.array(data[:, ind_numer], dtype=float)

features = np.concatenate([data_numer, data_cat], axis=1)
print(f'features type: {features.dtype}')

# split data into privileged features and normal features
# privileged features: age, race, sex
# normal features: the rest
ind_pf = [0, 10, 11]
print(f'Number of privileged feature: {len(ind_pf)}')
ind_nf = [i for i in range(13) if i not in ind_pf]
print(f'Number of normal features: {len(ind_nf)}')

# we work on binary classification for now
# >50K (1) or <=50K (0).
trans_label = {'<=50K': 0, '>50K': 1}
labels = np.array([trans_label[d] for d in data[:, ind_label].reshape(-1).tolist()], dtype='int')
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
    Path(f'../data/adult/fold{i}').mkdir(parents=True, exist_ok=True)
    # privileged and normal features
    dump_svmlight_file(X[train_index][:, ind_pf], y[train_index], \
                       f'../data/adult/fold{i}/adult.txt.pf.train')
    dump_svmlight_file(X[train_index][:, ind_nf], y[train_index], \
                       f'../data/adult/fold{i}/adult.txt.nf.train')

    dump_svmlight_file(X[valid_index][:, ind_pf], y[valid_index], \
                       f'../data/adult/fold{i}/adult.txt.pf.valid')
    dump_svmlight_file(X[valid_index][:, ind_nf], y[valid_index], \
                       f'../data/adult/fold{i}/adult.txt.nf.valid')

    dump_svmlight_file(X[test_index][:, ind_pf], y[test_index], \
                       f'../data/adult/fold{i}/adult.txt.pf.test')
    dump_svmlight_file(X[test_index][:, ind_nf], y[test_index], \
                       f'../data/adult/fold{i}/adult.txt.nf.test')
    # all features
    dump_svmlight_file(X[train_index], y[train_index], \
                       f'../data/adult/fold{i}/adult.txt.train')
    dump_svmlight_file(X[valid_index], y[valid_index], \
                       f'../data/adult/fold{i}/adult.txt.valid')
    dump_svmlight_file(X[test_index], y[test_index], \
                       f'../data/adult/fold{i}/adult.txt.test')

