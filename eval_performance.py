import numpy as np
from statistics import mean, stdev
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import (
        accuracy_score, 
        roc_auc_score, 
        precision_score, 
        recall_score
        )
from pathlib import Path
from common import all_feature_types, data_thred
from all_models import load_and_check_model
from utils import get_prediction, transfer_to_label, get_tpr

model_name = 'pi*' # 'normal', 'pi', 'pi*'
num_folds = 10 # the number of cross-validation folds
   
# general setting
dataset = 'diabetes_gender'
do_pred = True

# nf data
path_nf_trains = [f'data/{dataset}/fold{kf}/{dataset}.txt.nf.train' \
        for kf in range(num_folds)]
path_nf_valids = [f'data/{dataset}/fold{kf}/{dataset}.txt.nf.valid' \
        for kf in range(num_folds)]
path_nf_tests = [f'data/{dataset}/fold{kf}/{dataset}.txt.nf.test' \
        for kf in range(num_folds)]
feature_types_nf = all_feature_types[dataset]['nf']
# pf data
path_pf_trains = [f'data/{dataset}/fold{kf}/{dataset}.txt.pf.train' \
        for kf in range(num_folds)]
path_pf_valids = [f'data/{dataset}/fold{kf}/{dataset}.txt.pf.valid' \
        for kf in range(num_folds)]
path_pf_tests = [f'data/{dataset}/fold{kf}/{dataset}.txt.pf.test' \
        for kf in range(num_folds)]
feature_types_pf = all_feature_types[dataset]['pf']

if model_name == 'normal':
    path_models = [f'saved_models/{dataset}_nf_bst{kf}.json' for kf in range(num_folds)]

elif model_name == 'pi':
    path_models = [f'saved_models/{dataset}_np_bst{kf}.json' for kf in range(num_folds)]

elif model_name == 'pi*':
    path_models = [f'saved_models/{dataset}_nf_pf_bst{kf}.json' \
            for kf in range(num_folds)] 

Path(f'exp/{dataset}/{"pi_star" if model_name == "pi*" else model_name}')\
        .mkdir(parents=True, exist_ok=True)

all_names = ['AUC-ROC', 'Accuracy', 'TPR', 'Precision', 'Recall']
res_names = all_names[-2:]
results = [[] for _ in range(len(res_names))]

for kf in range(num_folds):
    if do_pred:
        # load saved model
        model = load_and_check_model(path_nf_trains[kf],
                                     path_nf_tests[kf],
                                     path_models[kf],
                                     feature_types=feature_types_nf)
        # predict probabilities
        preds = get_prediction(path_nf_tests[kf],
                               model,
                               feature_types=feature_types_nf)
        # save predict probabilities
        np.save(f'exp/{dataset}/'
                f'{"pi_star" if model_name == "pi*" else model_name}'
                f'/preds.npy', preds)

        # predict classes
        pred_classes = transfer_to_label(preds, 
                thred=data_thred[dataset][model_name])
        # save predict classes
        np.save(f'exp/{dataset}/'
                f'{"pi_star" if model_name == "pi*" else model_name}'
                f'/pred_classes.npy', pred_classes)
    else:
        preds = np.load(f'exp/{dataset}/'
                        f'{"pi_star" if model_name == "pi*" else model_name}'
                        f'/preds.npy')
        pred_classes = np.load(f'exp/{dataset}/'
                               f'{"pi_star" if model_name == "pi*" else model_name}'
                               f'/pred_classes.npy')

    _, y = load_svmlight_file(path_nf_tests[kf])
    # AUC-ROC
    if 'AUC-ROC' in res_names:
        results[res_names.index('AUC-ROC')].append(roc_auc_score(y, preds))

    # Accuracy
    if 'Accuracy' in res_names:
        results[res_names.index('Accuracy')].append(accuracy_score(y, pred_classes))

    # TPR
    if 'TPR' in res_names:
        results[res_names.index('TPR')].append(get_tpr(y, pred_classes))
    
    # Precision
    if 'Precision' in res_names:
        results[res_names.index('Precision')].append(precision_score(y, pred_classes))

    # Recall
    if 'Recall' in res_names:
        results[res_names.index('Recall')].append(recall_score(y, pred_classes))

print('Mean and standard deviation:')
for i in range(len(res_names)):
    print(f'{res_names[i]}')
    print(f'  Mean: {mean(results[i])}')
    print(f'  Std: {stdev(results[i])}')
