import numpy as np
from all_models import load_and_check_model, train_and_save_model
from all_models_cv import (
        train_and_save_model_cv, 
        train_and_save_model_with_pf_cv, 
        train_and_save_model_nf_pf_cv
    )
from common import (
        parameters, 
        models, 
        predictions, 
        all_feature_types, 
        all_feature_types_all
    )
from utils import get_prediction, argmax_r

# for now, only work on binary classification problem
# find best parameters setting for methods, according to the AUC-ROC performance
# on validation data across 10 folds, and report the performance on test data
thred_range = np.arange(0.001, 1.0, 0.001).tolist()
alpha_range = np.arange(0.001, 1.0, 0.001).tolist()
#alpha_range = np.arange(0.1, 1.0, 0.1).tolist()

model_name = 'all' # 'normal', 'pi', 'pi*', 'all'
num_folds = 10 # the number of cross-validation folds
   
# general setting
num_tree = 20
early_stopping_rounds = 5
early_stopping_method = 'aucroc'
dataset = 'numom2b_b'
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

# find best parameter setting
if model_name == 'normal':
    # nf only, the tuned parameter is `threshold` for accuracy
    # from 0.001 to 0.999 with interval 0.001, train once and 
    # check on validation for all threshold, find the one with
    # largest accuracy on validation data across all folds
    path_nf_models = [f'saved_models/{dataset}_nf_bst{kf}.json' for kf in range(num_folds)]
    for kf in range(num_folds):
        # train and save a model with normal features
        # results only need to contain validation and testing result
        results = train_and_save_model_cv(
                      path_nf_trains[kf],
                      path_nf_tests[kf],
                      path_nf_models[kf],
                      path_valid=path_nf_valids[kf],
                      thred_range=thred_range,
                      num_tree=num_tree,
                      early_stopping_rounds=early_stopping_rounds,
                      early_stopping_method=early_stopping_method,
                      feature_types=feature_types_nf)
        if kf == 0:
            results_accum = results
        else:
            for d in results:
                new_r = [(r1[0], [r12 + r22 for r12, r22 in zip(r1[1], r2[1])]) \
                        for r1, r2 in zip(results_accum[d], results[d])]
                results_accum[d] = new_r

    for d in results_accum:
        avg_r = [(r[0], [r1 / num_folds for r1 in r[1]]) for r in results_accum[d]]
        results_accum[d] = avg_r

    # print results under the best parameters setting
    print('-'*15)
    print('The best parameters setting:')

    dt = results_accum['dvalid']
    indices = [-1] # the 0-th is dummy
    for i in range(1, len(dt)):
        if dt[i][0] == 'Recall':
            # last occurrence of the max value
            ind = argmax_r(dt[i][1])
        else:
            ind = np.argmax(dt[i][1])
        print(f'Threshold for {dt[i][0]}: {thred_range[ind]}')
        indices.append(ind)

    print('The result under the best parameters setting:')
    for d in results_accum:
        rs = results_accum[d]
        print(f'{d}, {rs[0][0]}, {-rs[0][1][0]}')
        for i in range(1, len(rs)):
            r = rs[i] 
            ind = indices[i]
            print(f'{d}, {r[0]}, {r[1][ind]}')

elif model_name == 'pi':
    path_pf_models = [f'saved_models/{dataset}_pf_bst{kf}.json' for kf in range(num_folds)]
    path_np_models = [f'saved_models/{dataset}_np_bst{kf}.json' for kf in range(num_folds)]
    # find best parameter setting for alpha 
    for kf in range(num_folds):
        # train a boosting model with only privileged features
        train_and_save_model(path_pf_trains[kf],
                         path_pf_tests[kf],
                         path_pf_models[kf],
                         path_valid=path_pf_valids[kf],
                         num_tree=num_tree,
                         early_stopping_rounds=early_stopping_rounds,
                         early_stopping_method=early_stopping_method,
                         feature_types=feature_types_pf)

        # then use that model to provide more fine-grained probability distribution
        # for each instance when train model with normal features
        # check whether the performance is good
        # provide custom objective and metric again after the model is loaded
        models['pf_model'] = load_and_check_model(path_pf_trains[kf],
                                              path_pf_tests[kf],
                                              path_pf_models[kf],
                                              feature_types=feature_types_pf)
        predictions['pf_model'] = get_prediction(path_pf_trains[kf],
                                             models['pf_model'],
                                             feature_types=feature_types_pf)
        results_alpha = None
        for alpha in alpha_range:
            parameters['alpha'] = alpha
            results = train_and_save_model_with_pf_cv(
                                 path_nf_trains[kf], 
                                 path_nf_tests[kf], 
                                 path_np_models[kf], 
                                 path_valid=path_nf_valids[kf], 
                                 num_tree=num_tree,
                                 early_stopping_rounds=early_stopping_rounds,
                                 early_stopping_method=early_stopping_method,
                                 feature_types=feature_types_nf)
            if results_alpha is None:
                results_alpha = results
            else:
                for d in results:
                    for r1, r2 in zip(results_alpha[d], results[d]):
                        r1[1].append(r2[1][0])
        if kf == 0:
            results_accum = results_alpha
        else:
            for d in results_alpha:
                new_r = [(r1[0], [r12 + r22 for r12, r22 in zip(r1[1], r2[1])]) \
                        for r1, r2 in zip(results_accum[d], results_alpha[d])]
                results_accum[d] = new_r

    for d in results_accum:
        # remember the AUCROC has '-'
        avg_r = [(r[0], [-r1 / num_folds for r1 in r[1]]) for r in results_accum[d]]
        results_accum[d] = avg_r

    ind = np.argmax(results_accum['dvalid'][0][1])
    best_alpha = alpha_range[ind]
    print(f'Best value of alpha: {best_alpha}')

    results_best_alpha = [(d, results_accum[d][0][0], results_accum[d][0][1][ind]) \
                         for d in results_accum]
   
    
    # print results under the best parameters setting
    parameters['alpha'] = best_alpha
    for kf in range(num_folds):
        # train a boosting model with only privileged features
        train_and_save_model(path_pf_trains[kf],
                         path_pf_tests[kf],
                         path_pf_models[kf],
                         path_valid=path_pf_valids[kf],
                         num_tree=num_tree,
                         early_stopping_rounds=early_stopping_rounds,
                         early_stopping_method=early_stopping_method,
                         feature_types=feature_types_pf)

        # then use that model to provide more fine-grained probability distribution
        # for each instance when train model with normal features
        # check whether the performance is good
        # provide custom objective and metric again after the model is loaded
        models['pf_model'] = load_and_check_model(path_pf_trains[kf],
                                              path_pf_tests[kf],
                                              path_pf_models[kf],
                                              feature_types=feature_types_pf)
        predictions['pf_model'] = get_prediction(path_pf_trains[kf],
                                             models['pf_model'],
                                             feature_types=feature_types_pf)
        results = train_and_save_model_with_pf_cv(
                             path_nf_trains[kf], 
                             path_nf_tests[kf], 
                             path_np_models[kf], 
                             path_valid=path_nf_valids[kf], 
                             thred_range=thred_range,
                             num_tree=num_tree,
                             early_stopping_rounds=early_stopping_rounds,
                             early_stopping_method=early_stopping_method,
                             feature_types=feature_types_nf)
        if kf == 0:
            results_accum = results
        else:
            for d in results:
                new_r = [(r1[0], [r12 + r22 for r12, r22 in zip(r1[1], r2[1])]) \
                        for r1, r2 in zip(results_accum[d], results[d])]
                results_accum[d] = new_r
                
    for d in results_accum:
        avg_r = [(r[0], [r1 / num_folds for r1 in r[1]]) for r in results_accum[d]]
        results_accum[d] = avg_r

    print('-'*15)
    print('The best parameters setting:')
    
    print(f'Best value of alpha: {best_alpha}')

    dt = results_accum['dvalid']
    indices = [-1] # the 0-th is dummy
    for i in range(1, len(dt)):
        if dt[i][0] == 'Recall':
            # last occurrence of the max value
            ind = argmax_r(dt[i][1])
        else:
            ind = np.argmax(dt[i][1])
        print(f'Threshold for {dt[i][0]}: {thred_range[ind]}')
        indices.append(ind)

    print('The result under the best parameters setting:')
    for d in results_accum:
        rs = results_accum[d]
        print(f'{d}, {rs[0][0]}, {-rs[0][1][0]}')

        for i in range(1, len(rs)):
            r = rs[i]
            ind = indices[i]
            print(f'{d}, {r[0]}, {r[1][ind]}')
    
    print('Sanity check:')
    for d, r1, r2 in results_best_alpha:
        print(f'{d}, {r1}, {r2}')
    

elif model_name == 'pi*':
    path_nf_pf_models = [f'saved_models/{dataset}_nf_pf_bst{kf}.json' for kf in range(num_folds)] 
    # find best parameter setting for alpha
    for kf in range(num_folds):
        results_alpha = None
        for alpha in alpha_range:
            parameters['alpha'] = alpha
            # Co-ordinate gradient descent training of nf and pf model
            results = train_and_save_model_nf_pf_cv(
                               path_nf_train=path_nf_trains[kf], 
                               path_nf_test=path_nf_tests[kf], 
                               path_pf_train=path_pf_trains[kf], 
                               path_pf_test=path_pf_tests[kf], 
                               model_file=path_nf_pf_models[kf], 
                               path_nf_valid=path_nf_valids[kf], 
                               path_pf_valid=path_pf_valids[kf],
                               num_tree=num_tree,
                               early_stopping_rounds=early_stopping_rounds,
                               early_stopping_method=early_stopping_method,
                               feature_types_nf=feature_types_nf,
                               feature_types_pf=feature_types_pf,
                               verbose=True)
            if results_alpha is None:
                results_alpha = results
            else:
                for d in results:
                    for r1, r2 in zip(results_alpha[d], results[d]):
                        r1[1].append(r2[1][0])
        if kf == 0:
            results_accum = results_alpha
        else:
            for d in results_alpha:
                new_r = [(r1[0], [r12 + r22 for r12, r22 in zip(r1[1], r2[1])]) \
                        for r1, r2 in zip(results_accum[d], results_alpha[d])]
                results_accum[d] = new_r

    for d in results_accum:
        # here the AUCROC has no '-'
        avg_r = [(r[0], [r1 / num_folds for r1 in r[1]]) for r in results_accum[d]]
        results_accum[d] = avg_r

    ind = np.argmax(results_accum['dvalid'][0][1])
    best_alpha = alpha_range[ind]
    print(f'Best value of alpha: {best_alpha}')
    
    results_best_alpha = [(d, results_accum[d][0][0], results_accum[d][0][1][ind]) \
                         for d in results_accum]
    
    # print results under the best parameters setting
    parameters['alpha'] = best_alpha
    for kf in range(num_folds):
        # Co-ordinate gradient descent training of nf and pf model
        results = train_and_save_model_nf_pf_cv(
                           path_nf_train=path_nf_trains[kf], 
                           path_nf_test=path_nf_tests[kf], 
                           path_pf_train=path_pf_trains[kf], 
                           path_pf_test=path_pf_tests[kf], 
                           model_file=path_nf_pf_models[kf], 
                           path_nf_valid=path_nf_valids[kf], 
                           path_pf_valid=path_pf_valids[kf],
                           thred_range=thred_range, # thred range for accuracy
                           num_tree=num_tree,
                           early_stopping_rounds=early_stopping_rounds,
                           early_stopping_method=early_stopping_method,
                           feature_types_nf=feature_types_nf,
                           feature_types_pf=feature_types_pf,
                           verbose=True)
        if kf == 0:
            results_accum = results
        else:
            for d in results:
                new_r = [(r1[0], [r12 + r22 for r12, r22 in zip(r1[1], r2[1])]) \
                        for r1, r2 in zip(results_accum[d], results[d])]
                results_accum[d] = new_r

    for d in results_accum:
        avg_r = [(r[0], [r1 / num_folds for r1 in r[1]]) for r in results_accum[d]]
        results_accum[d] = avg_r

    print('-'*15)
    print('The best parameters setting:')

    print(f'Best value of alpha: {best_alpha}')
    dt = results_accum['dvalid']
    indices = [-1] # the 0-th is dummy
    for i in range(1, len(dt)):
        if dt[i][0] == 'Recall':
            # last occurrence of the max value
            ind = argmax_r(dt[i][1])
        else:
            ind = np.argmax(dt[i][1])
        print(f'Threshold for {dt[i][0]}: {thred_range[ind]}')
        indices.append(ind)

    print('The result under the best parameters setting:')
    for d in results_accum:
        rs = results_accum[d]
        print(f'{d}, {rs[0][0]}, {rs[0][1][0]}')

        for i in range(1, len(rs)):
            r = rs[i]
            ind = indices[i]
            print(f'{d}, {r[0]}, {r[1][ind]}')
    
    print('Sanity check:')
    for d, r1, r2 in results_best_alpha:
        print(f'{d}, {r1}, {r2}')
    
elif model_name == 'all':
    # all features = normal features + privileged features
    # only used to show upper bound of fairness metrics
    path_all_trains = [f'data/{dataset}/fold{kf}/{dataset}.txt.train' \
                       for kf in range(num_folds)]
    path_all_valids = [f'data/{dataset}/fold{kf}/{dataset}.txt.valid' \
                      for kf in range(num_folds)]
    path_all_tests = [f'data/{dataset}/fold{kf}/{dataset}.txt.test' \
                     for kf in range(num_folds)]
    feature_types_all = all_feature_types_all[dataset]['all']
    path_all_models = [f'saved_models/{dataset}_all_bst{kf}.json' for kf in range(num_folds)]
    for kf in range(num_folds):
        # train and save a model with normal features
        # results only need to contain validation and testing result
        results = train_and_save_model_cv(
                      path_all_trains[kf],
                      path_all_tests[kf],
                      path_all_models[kf],
                      path_valid=path_all_valids[kf],
                      thred_range=thred_range,
                      num_tree=num_tree,
                      early_stopping_rounds=early_stopping_rounds,
                      early_stopping_method=early_stopping_method,
                      feature_types=feature_types_all)
        if kf == 0:
            results_accum = results
        else:
            for d in results:
                new_r = [(r1[0], [r12 + r22 for r12, r22 in zip(r1[1], r2[1])]) \
                        for r1, r2 in zip(results_accum[d], results[d])]
                results_accum[d] = new_r

    for d in results_accum:
        avg_r = [(r[0], [r1 / num_folds for r1 in r[1]]) for r in results_accum[d]]
        results_accum[d] = avg_r

    # print results under the best parameters setting
    print('-'*15)
    print('The best parameters setting:')

    dt = results_accum['dvalid']
    indices = [-1] # the 0-th is dummy
    for i in range(1, len(dt)):
        if dt[i][0] == 'Recall':
            # last occurrence of the max value
            ind = argmax_r(dt[i][1])
        else:
            ind = np.argmax(dt[i][1])
        print(f'Threshold for {dt[i][0]}: {thred_range[ind]}')
        indices.append(ind)

    print('The result under the best parameters setting:')
    for d in results_accum:
        rs = results_accum[d]
        print(f'{d}, {rs[0][0]}, {-rs[0][1][0]}')
        for i in range(1, len(rs)):
            r = rs[i] 
            ind = indices[i]
            print(f'{d}, {r[0]}, {r[1][ind]}')
