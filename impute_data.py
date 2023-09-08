import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

# general setting
num_folds = 10 # the number of cross-validation folds

dataset = 'numom2b_b'
ind_pf = [17, 18]

# data with all features
path_trains = [f'data/{dataset}/fold{kf}/{dataset}.txt.train' \
               for kf in range(num_folds)]
path_tests = [f'data/{dataset}/fold{kf}/{dataset}.txt.test' \
              for kf in range(num_folds)]

path_tests_imputed = [f'data/{dataset}/fold{kf}/{dataset}.txt.imputed.test' \
                      for kf in range(num_folds)]

for kf in range(num_folds):
    X_train, _ = load_svmlight_file(path_trains[kf])
    X_train = X_train.toarray()
    X, y = load_svmlight_file(path_tests[kf])
    X = X.toarray()

    if dataset == 'numom2b_b': # special process
        X[:, 0] = 1
        for i in range(1, 8):
            X[:, i] = 0
    else:
        for i in ind_pf:
            values, counts = np.unique(X_train[:, i], return_counts=True)
            ind = np.argmax(counts)
            X[:, i] = values[ind] # the most frequent element
    dump_svmlight_file(X, y, path_tests_imputed[kf])

