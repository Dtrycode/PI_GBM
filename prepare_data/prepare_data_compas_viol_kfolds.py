import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path


path = '../data/compas-scores-two-years-violent.csv'
dataset = 'compas_viol'
data = pd.read_csv(path, sep=',', header=0, index_col=0)

print(data.head())
print(f'Number of instances loaded: {len(data)}')

# filter over 'days_b_screening_arrest'
data = data[(data['days_b_screening_arrest'] >= -30) & (data['days_b_screening_arrest'] <= 30)]
print(f'Number of instances left after filtering over days_b_screening_arrest: {len(data)}')

# select columns
names = ['sex', 'age', 'age_cat', 'race', \
         'juv_fel_count', 'juv_misd_count', 'juv_other_count', \
         'priors_count', 'c_charge_degree', \
         'score_text', 'v_score_text', \
         'two_year_recid']

data = data[names]

# add together three 'juv_' columns
juvs = data['juv_fel_count'] + data['juv_misd_count'] + data['juv_other_count']
data = data.drop(columns=['juv_fel_count', 'juv_misd_count', 'juv_other_count'])
data.insert(4, 'juv_crime', juvs)

data = data.dropna() # there are 5 rows [3708, 4995, 6283, 7132, 8766] 
# with nan 'score_text' which are not in not viol
print(f'Number of instances left after dropping rows with missing value: {len(data)}')

data = data.to_numpy()

# categorical features
ind_cat = [0, 2, 3, 6, 7, 8]
# numeric features
ind_numer = [1, 4, 5]
# labels
ind_label = [9]

data_cat = data[:, ind_cat]
enc = OrdinalEncoder()
data_cat = enc.fit_transform(data_cat)

data_numer = np.array(data[:, ind_numer], dtype=float)

features = np.concatenate([data_numer, data_cat], axis=1)
print(f'features type: {features.dtype}')

# split data into privileged features and normal features
# privileged features: sex, race
# normal features: the rest
ind_pf = [3, 5] # sex, race
print(f'Number of privileged feature: {len(ind_pf)}')
ind_nf = [i for i in range(9) if i not in ind_pf+[0]] # not include 'age'
print(f'Number of normal features: {len(ind_nf)}')

# we work on binary classification for now
# 1 (1) or 0 (0).
trans_label = {0: 0, 1: 1}
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
    Path(f'../data/{dataset}/fold{i}').mkdir(parents=True, exist_ok=True)
    # privileged and normal features
    dump_svmlight_file(X[train_index][:, ind_pf], y[train_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.pf.train')
    dump_svmlight_file(X[train_index][:, ind_nf], y[train_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.nf.train')

    dump_svmlight_file(X[valid_index][:, ind_pf], y[valid_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.pf.valid')
    dump_svmlight_file(X[valid_index][:, ind_nf], y[valid_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.nf.valid')

    dump_svmlight_file(X[test_index][:, ind_pf], y[test_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.pf.test')
    dump_svmlight_file(X[test_index][:, ind_nf], y[test_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.nf.test')
    # all features
    dump_svmlight_file(X[train_index], y[train_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.train')
    dump_svmlight_file(X[valid_index], y[valid_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.valid')
    dump_svmlight_file(X[test_index], y[test_index], \
                       f'../data/{dataset}/fold{i}/{dataset}.txt.test')

