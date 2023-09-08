import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path


path = '../data/diabetic_data.csv'
dataset = 'diabetes_gender'
data = pd.read_csv(path, header=0)
print(data.head())
print(f'Number of instances loaded: {len(data)}')

# all features and label
names = ['race', 'gender', 'age', 'time_in_hospital', 'num_procedures', \
        'num_medications', 'number_outpatient', 'number_emergency', \
        'number_inpatient', 'A1Cresult', 'metformin', 'chlorpropamide', \
        'glipizide', 'rosiglitazone', 'acarbose', 'miglitol', 'diabetesMed', \
        'readmitted']

data = data[names]

# remove rows with missing value of 'race' or 'readmitted' is NO
data = data[(data['race'] != '?') & (data['readmitted'] != 'NO')]
print(f'Number of instances left: {len(data)}')

# age to numeric
# ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', 
#  '[60-70)', '[70-80)', '[80-90)', '[90-100)']
data['age'] = data['age'].map(
        {s: i for i, s in enumerate(['[0-10)', '[10-20)', '[20-30)', \
              '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', \
              '[80-90)', '[90-100)'])})

data = data.to_numpy()

# categorical features
# race, gender, A1Cresult, metformin, chlorpropamide, glipizide, 
# rosiglitazone, acarbose, miglitol, diabetesMed
ind_cat = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16]
# numeric features
# age, time_in_hospital, num_procedures, num_medications, number_outpatient, 
# number_emergency, number_inpatient
ind_numer = list(range(2, 9))
# labels
# readmitted
ind_label = [17]

data_cat = data[:, ind_cat]
enc = OrdinalEncoder()
data_cat = enc.fit_transform(data_cat)

data_numer = np.array(data[:, ind_numer], dtype=float)

features = np.concatenate([data_numer, data_cat], axis=1)
print(f'features type: {features.dtype}')

# split data into privileged features and normal features
# privileged features: gender
# normal features: the rest
ind_pf = [8] # gender
#ind_pf = [0, 7, 8] # age, race, gender
print(f'Number of privileged feature: {len(ind_pf)}')
ind_nf = [i for i in range(17) if i not in ind_pf]
print(f'Number of normal features: {len(ind_nf)}')

# we work on binary classification for now
# <30 (1) or >30 (0).
trans_label = {'>30': 0, '<30': 1}
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

