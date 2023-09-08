import numpy as np
from sklearn.datasets import load_svmlight_file
from common import (
        all_feature_types, 
        all_feature_types_all, 
        data_thred,
        data_protect_group
        )
from all_models import load_and_check_model
from utils import (
        get_prediction, 
        transfer_to_label, 
        mask_adult, 
        mask_credit, 
        mask_numom2b_b,
        get_tpr
        )
from metrics import (
        delta_eo, 
        delta_dp, 
        calculate_eo, 
        calculate_sp, 
        calculate_abroca
        )

# for now, only work on binary classification problem
# find best parameters setting for methods, according to the AUC-ROC performance
# on validation data across 10 folds, and report the performance on test data

model_name = 'all' # 'normal', 'pi', 'pi*', 'all'
num_folds = 10 # the number of cross-validation folds
   
# general setting
dataset = 'rare'
all_metrics = ['Equalized Odds', 'Statistical Parity', 'Absolute Between-ROC Area', \
        'TPR']
eval_metrics = all_metrics[:-1]

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

elif model_name == 'all':
    # data with all features
    path_nf_trains = [f'data/{dataset}/fold{kf}/{dataset}.txt.train' \
                      for kf in range(num_folds)]
    path_nf_valids = [f'data/{dataset}/fold{kf}/{dataset}.txt.valid' \
                      for kf in range(num_folds)]
    # can only use imputed testing data
    path_nf_tests = [f'data/{dataset}/fold{kf}/{dataset}.txt.imputed.test' \
                     for kf in range(num_folds)]
    feature_types_nf = all_feature_types_all[dataset]['all']
    path_models = [f'saved_models/{dataset}_all_bst{kf}.json' \
                   for kf in range(num_folds)]

for kf in range(num_folds):
    # load saved model
    model = load_and_check_model(path_nf_trains[kf],
                                path_nf_tests[kf],
                                 path_models[kf],
                                 feature_types=feature_types_nf)
    # predict probabilities
    preds = get_prediction(path_nf_tests[kf],
                           model,
                           feature_types=feature_types_nf)

    # predict classes
    pred_classes = transfer_to_label(preds, 
            thred=data_thred[dataset][model_name])

    X, y = load_svmlight_file(path_pf_tests[kf])
    X = X.toarray()

    if dataset == 'adult':
        X = mask_adult(X)
    elif dataset == 'credit':
        X = mask_credit(X)
    elif dataset == 'numom2b_b':
        X = mask_numom2b_b(X)
    
    # equalized odds
    name_eo = 'Equalized Odds'
    if name_eo in eval_metrics:
        results_eo = calculate_eo(X, y, pred_classes, 
                data_protect_group[dataset])
        results_eo = [r[name_eo] for r in results_eo]
    
    # statistical parity
    name_sp = 'Statistical Parity'
    if name_sp in eval_metrics:
        results_sp = calculate_sp(X, y, pred_classes, 
                data_protect_group[dataset])
        results_sp = [abs(r[name_sp]) for r in results_sp]

    # Absolute Between-ROC Area
    name_abroca = 'Absolute Between-ROC Area'
    if name_abroca in eval_metrics:
        results_abroca = calculate_abroca(X, y, preds, data_protect_group[dataset], 
                n_grid=10000, plot_slices=False, filename=f'{dataset}.abroca.pdf')

    # True Positive Rate (only to calculate)
    name_tpr = 'TPR'
    if name_tpr in eval_metrics:
        results_tpr = [get_tpr(y, pred_classes)]

    if kf == 0:
        results = {}
        if name_eo in eval_metrics:
            results[name_eo] = results_eo
        if name_sp in eval_metrics:
            results[name_sp] = results_sp
        if name_abroca in eval_metrics:
            results[name_abroca] = results_abroca
        if name_tpr in eval_metrics:
            results[name_tpr] = results_tpr
    else:
        if name_eo in eval_metrics:
            results[name_eo] = [r0 + r1 for r0, r1 in zip(results[name_eo], results_eo)]
        if name_sp in eval_metrics:
            results[name_sp] = [r0 + r1 for r0, r1 in zip(results[name_sp], results_sp)]
        if name_abroca in eval_metrics:
            results[name_abroca] = [r0 + r1 for r0, r1 in \
                    zip(results[name_abroca], results_abroca)]
        if name_tpr in eval_metrics:
            results[name_tpr] = [r0 + r1 for r0, r1 in zip(results[name_tpr], results_tpr)]

print('Fairness metrics evaluation:')
for name in results:
    results[name] = [round(v / num_folds, 3) for v in results[name]]
    print(f'{name}: {results[name]}')
