import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from utils import (
        print_accuracy, 
        get_accuracy, 
        get_precision, 
        get_recall, 
        get_f1,
        sigmoid
        )
from metrics import acc, aucpr, aucroc, make_loss
from objectives import neg_logl, neg_logl_np
from all_models import get_early_stopping_method
from common import models, parameters, predictions
import warnings

#suppress warnings
warnings.filterwarnings('ignore')


# functions to train, save and load model
def train_and_save_model_cv(path_train, 
                            path_test, 
                            model_file, 
                            path_valid=None, 
                            thred_range=None,
                            num_tree=5,
                            early_stopping_rounds=1,
                            early_stopping_method='aucpr',
                            max_depth=6,
                            learning_rate=0.3,
                            max_cat_to_onehot=5,
                            feature_types=None,
                            fair_metric=None):
    '''
    Arguments:
      path_train: path of train data
      path_test: path of test data
      model_file: file path to save trained model
    '''
    # validation data must be provided for tuning parameters
    if path_valid is None:
        print(f'Error: validation data must be provided for tuning parameters.')
        exit(0)

    es_name, es_method = get_early_stopping_method(early_stopping_method)

    if feature_types:
        dtrain = xgb.DMatrix(path_train, feature_types=feature_types, enable_categorical=True)
        dvalid = xgb.DMatrix(path_valid, feature_types=feature_types, enable_categorical=True)
        dtest = xgb.DMatrix(path_test, feature_types=feature_types, enable_categorical=True)
        tree_method = 'approx'
        tree_parameters = {'tree_method': tree_method, 'seed': 1994,
                           'disable_default_eval_metric': 1,
                           'max_depth': max_depth, # maximum tree depth
                           'eta': learning_rate, # learning rate
                           'max_cat_to_onehot': max_cat_to_onehot 
                               # number of categories threshold for one-hot
                          }
    else:
        dtrain = xgb.DMatrix(path_train)
        dvalid = xgb.DMatrix(path_valid)
        dtest = xgb.DMatrix(path_test)
        tree_method='exact'
        tree_parameters = {'tree_method': tree_method, 'seed': 1994,
                           'disable_default_eval_metric': 1,
                           'max_depth': max_depth, # maximum tree depth
                           'eta': learning_rate # learning rate
                          }

    data = [(dtrain, 'dtrain'), (dtest, 'dtest'), (dvalid, 'dvalid')]

    results = {}
    bst = xgb.train(tree_parameters,
                    dtrain=dtrain,
                    num_boost_round=num_tree,
                    obj=neg_logl,
                    custom_metric=make_loss(es_method), # feval always gets raw prediction, 
                                     # custom_metric gets transformed prediction 
                                     # if not use custom objective
                    evals=data,
                    early_stopping_rounds=early_stopping_rounds, 
                                                 # works on loss function, not score function
                                                 # loss decreases
                    evals_result=results)
    # results to return
    results_bst = {}
    for d in results:
        r = list(results[d].items())[0]
        results_bst[d] = [(r[0], [r[1][bst.best_iteration]])]

    # find the turning point from validation
    print('Best iteration\tBest score\tBest number of trees:')
    print(f'{bst.best_iteration}\t{bst.best_score}\t{bst.best_ntree_limit}') 
    print(f'Number of trees learned: {bst.num_boosted_rounds()}')
    # save trained model
    print(f'Slicing {bst.best_iteration+1} trees (best) from {bst.num_boosted_rounds()} trees')
    bst = bst[:bst.best_iteration+1]
    bst.save_model(model_file)
    print(f'Saved model {model_file}')

    if thred_range:
        # accuracy
        results_accuracy = get_accuracy(data, bst, thred_range)
        for d in results_accuracy:
            results_bst[d].append(('Accuracy', results_accuracy[d]))
        # precision
        results_precision = get_precision(data, bst, thred_range)
        for d in results_precision:
            results_bst[d].append(('Precision', results_precision[d]))

        # recall
        results_recall = get_recall(data, bst, thred_range)
        for d in results_recall:
            results_bst[d].append(('Recall', results_recall[d]))

        # f1
        results_f1 = get_f1(data, bst, thred_range)
        for d in results_f1.keys():
            results_bst[d].append(('F1', results_f1[d]))

    return results_bst


def train_and_save_model_with_pf_cv(path_train, 
                                 path_test, 
                                 model_file, 
                                 path_valid=None, 
                                 thred_range=None,
                                 num_tree=5,
                                 early_stopping_rounds=1,
                                 early_stopping_method='aucpr',
                                 max_depth=6,
                                 learning_rate=0.3,
                                 max_cat_to_onehot=5,
                                 feature_types=None):
    ''' previledged model must be already loaded into `models` with name `pf_model`
    Arguments:
      path_train: path of train data
      path_test: path of test data
      model_file: file path to save trained model
    '''
    pf_model_not_loaded = 'pf_model' not in models
    pf_model_prediction_not_generated = 'pf_model' not in predictions
    alpha_not_given = 'alpha' not in parameters
    if pf_model_not_loaded or pf_model_prediction_not_generated or alpha_not_given:
        if pf_model_not_loaded:
            print(f'Error: previledged model not loaded.')
        if pf_model_prediction_not_generated:
            print(f'Error: previledged model prediction not generated.')
        if alpha_not_given:
            print(f'Error: alpha not given.')
        exit(0)

    # validation data must be provided for tuning parameters
    if path_valid is None:
        print(f'Error: validation data must be provided for tuning parameters.')
        exit(0)

    es_name, es_method = get_early_stopping_method(early_stopping_method)

    if feature_types:
        dtrain = xgb.DMatrix(path_train, feature_types=feature_types, enable_categorical=True)
        dvalid = xgb.DMatrix(path_valid, feature_types=feature_types, enable_categorical=True)
        dtest = xgb.DMatrix(path_test, feature_types=feature_types, enable_categorical=True)
        tree_method='approx'
        tree_parameters = {'tree_method': tree_method, 'seed': 1994,
                           'disable_default_eval_metric': 1,
                           'max_depth': max_depth, # maximum tree depth
                           'eta': learning_rate, # learning rate
                           'max_cat_to_onehot': max_cat_to_onehot
                               # number of categories threshold for one-hot
                          }
    else:
        dtrain = xgb.DMatrix(path_train)
        dvalid = xgb.DMatrix(path_valid)
        dtest = xgb.DMatrix(path_test)
        tree_method='exact'
        tree_parameters = {'tree_method': tree_method, 'seed': 1994,
                           'disable_default_eval_metric': 1,
                           'max_depth': max_depth, # maximum tree depth
                           'eta': learning_rate # learning rate
                          }

    data = [(dtrain, 'dtrain'), (dtest, 'dtest'), (dvalid, 'dvalid')]

    results = {}
    bst = xgb.train(tree_parameters,
                    dtrain=dtrain,
                    num_boost_round=num_tree,
                    obj=neg_logl_np,
                    custom_metric=make_loss(es_method), # feval always gets raw prediction, 
                                         # custom_metric gets transformed prediction 
                                         # if not use custom objective
                    evals=data,
                    early_stopping_rounds=early_stopping_rounds, 
                                             # works on loss function, not score function
                                             # loss decreases
                    evals_result=results)

    # results to return
    results_bst = {}
    for d in results.keys():
        r = list(results[d].items())[0]
        results_bst[d] = [(r[0], [r[1][bst.best_iteration]])]

    print('Best iteration\tBest score\tBest number of trees:')
    print(f'{bst.best_iteration}\t{bst.best_score}\t{bst.best_ntree_limit}')
    print(f'Number of trees learned: {bst.num_boosted_rounds()}')
    
    # save trained model
    print(f'Slicing {bst.best_iteration+1} trees (best) from {bst.num_boosted_rounds()} trees')
    bst = bst[:bst.best_iteration+1]
    bst.save_model(model_file)
    print(f'Saved model {model_file}')

    if thred_range:
        # accuracy
        results_accuracy = get_accuracy(data, bst, thred_range)
        for d in results_accuracy.keys():
            results_bst[d].append(('Accuracy', results_accuracy[d]))

        # precision
        results_precision = get_precision(data, bst, thred_range)
        for d in results_precision:
            results_bst[d].append(('Precision', results_precision[d]))

        # recall
        results_recall = get_recall(data, bst, thred_range)
        for d in results_recall:
            results_bst[d].append(('Recall', results_recall[d]))

        # f1
        results_f1 = get_f1(data, bst, thred_range)
        for d in results_f1.keys():
            results_bst[d].append(('F1', results_f1[d]))

    return results_bst


def train_and_save_model_nf_pf_cv(path_nf_train, path_nf_test, 
                               path_pf_train, path_pf_test, 
                               model_file, 
                               path_nf_valid=None, path_pf_valid=None,
                               thred_range=None,
                               num_tree=5,
                               early_stopping_rounds=1,
                               early_stopping_method='aucpr',
                               max_depth=6,
                               learning_rate=0.3,
                               max_cat_to_onehot=5,
                               feature_types_nf=None,
                               feature_types_pf=None,
                               verbose=False):
    ''' Co-ordinate gradient descent between normal features and privileged features
    Arguments:
      path_train: path of train data
      path_test: path of test data
      model_file: file path to save trained model
    '''
    # hyperparameter alpha should be given
    alpha_not_given = 'alpha' not in parameters
    if alpha_not_given:
        print(f'Error: alpha not given.')
        exit(0)

    # validation data must be provided for tuning parameters
    if path_nf_valid is None:
        print(f'Error: validation data normal features must be provided for tuning parameters.')
        exit(0)

    es_name, es_method = get_early_stopping_method(early_stopping_method)

    if feature_types_nf:
        dtrain_nf = xgb.DMatrix(path_nf_train, 
                                feature_types=feature_types_nf, 
                                enable_categorical=True)
        dvalid_nf = xgb.DMatrix(path_nf_valid,
                                feature_types=feature_types_nf,
                                enable_categorical=True)
        dtest_nf = xgb.DMatrix(path_nf_test, 
                               feature_types=feature_types_nf, 
                               enable_categorical=True)
        tree_method_nf = 'approx'
        tree_parameters_nf = {'tree_method': tree_method_nf, 'seed': 1994,
                              'disable_default_eval_metric': 1,
                              'max_depth': max_depth, # maximum tree depth
                              'eta': learning_rate, # learning rate
                              'max_cat_to_onehot': max_cat_to_onehot
                                  # number of categories threshold for one-hot
                             }
    else:
        dtrain_nf = xgb.DMatrix(path_nf_train)
        dvalid_nf = xgb.DMatrix(path_nf_valid)
        dtest_nf = xgb.DMatrix(path_nf_test)
        tree_method_nf = 'exact'
        tree_parameters_nf = {'tree_method': tree_method_nf, 'seed': 1994,
                              'disable_default_eval_metric': 1,
                              'max_depth': max_depth, # maximum tree depth
                              'eta': learning_rate # learning rate
                             }

    if feature_types_pf:
        dtrain_pf = xgb.DMatrix(path_pf_train, 
                                feature_types=feature_types_pf, 
                                enable_categorical=True)
        dvalid_pf = xgb.DMatrix(path_pf_valid,
                                feature_types=feature_types_pf,
                                enable_categorical=True)
        dtest_pf = xgb.DMatrix(path_pf_test, 
                               feature_types=feature_types_pf, 
                               enable_categorical=True)
        tree_method_pf = 'approx'
        tree_parameters_pf = {'tree_method': tree_method_pf, 'seed': 1994,
                              'disable_default_eval_metric': 1,
                              'max_depth': max_depth, # maximum tree depth
                              'eta': learning_rate, # learning rate
                              #'eta': 0.1, # learning rate
                              'max_cat_to_onehot': max_cat_to_onehot
                                  # number of categories threshold for one-hot
                             }
    else:
        dtrain_pf = xgb.DMatrix(path_pf_train)
        dvalid_pf = xgb.DMatrix(path_pf_valid)
        dtest_pf = xgb.DMatrix(path_pf_test)
        tree_method_pf = 'exact'
        tree_parameters_pf = {'tree_method': tree_method_pf, 'seed': 1994,
                              'disable_default_eval_metric': 1,
                              'max_depth': max_depth, # maximum tree depth
                              'eta': learning_rate # learning rate
                             }

    model_saved = False

    train_scores = []
    valid_scores = []
    test_scores = []
    nf_data = [(dtrain_nf, 'dtrain'), (dtest_nf, 'dtest'), (dvalid_nf, 'dvalid')]
    pf_data = [(dtrain_pf, 'dtrain'), (dtest_pf, 'dtest'), (dvalid_pf, 'dvalid')]

    # initialize pf model
    if verbose:
        print('Initialize pf model')
    results = {}
    bst_pf = xgb.train(tree_parameters_pf,
                        dtrain=dtrain_pf,
                        num_boost_round=0,
                        obj=neg_logl_np,
                        custom_metric=es_method, # feval always gets raw prediction, 
                                             # custom_metric gets transformed prediction 
                                             # if not use custom objective
                        evals=pf_data,
                        evals_result=results)

    bst_nf = None

    for i in range(num_tree):
        if verbose:
            print(f'Iteration {i}:')
        # for nf model, update `pf_model` and its predictions on train data
        models['pf_model'] = bst_pf
        predictions['pf_model'] = sigmoid(models['pf_model'].predict(dtrain_pf))
        # train one tree for nf model
        if verbose:
            print('Train one tree for nf model')
        results = {}
        bst_nf = xgb.train(tree_parameters_nf,
                     dtrain=dtrain_nf,
                     num_boost_round=1,
                     obj=neg_logl_np,
                     custom_metric=es_method, # feval always gets raw prediction, 
                                          # custom_metric gets transformed prediction 
                                          # if not use custom objective
                     evals=nf_data,
                     evals_result=results,
                     xgb_model=bst_nf # continue training
                     )
        train_scores.append(results['dtrain'][es_name][0])
        test_scores.append(results['dtest'][es_name][0])
        # validation and early stopping is applied
        if not valid_scores:
            # for early stopping with patience
            patience = early_stopping_rounds
            max_ind = 0
            # store the performance
            valid_scores.append(results['dvalid'][es_name][0])
        elif valid_scores[max_ind] < results['dvalid'][es_name][0]:
            patience = early_stopping_rounds
            max_ind = len(valid_scores)
            # store the performance
            valid_scores.append(results['dvalid'][es_name][0])
        elif patience > 1:
            patience -= 1
            # store the performance
            valid_scores.append(results['dvalid'][es_name][0])
        else:
            # slice the model and break loop
            print('Best iteration\tBest score\tBest number of trees:')
            print(f'{max_ind}\t{valid_scores[max_ind]}\t{max_ind+1}')
            print(f'Number of trees learned: {bst_nf.num_boosted_rounds()}')
    
            # save trained model
            print(f'Slicing {max_ind+1} trees (best) from {bst_nf.num_boosted_rounds()} trees')
            bst_nf = bst_nf[:max_ind+1]
            bst_nf.save_model(model_file)
            model_saved = True
            print(f'Saved model {model_file}')
            break

        # for pf model, update `pf_model` and its predictions on train data
        models['pf_model'] = bst_nf
        predictions['pf_model'] = sigmoid(models['pf_model'].predict(dtrain_nf))
        # train one tree of pf model
        if verbose:
            print('Train one tree for pf model')
        results = {}
        bst_pf = xgb.train(tree_parameters_pf,
                     dtrain=dtrain_pf,
                     num_boost_round=1,
                     obj=neg_logl_np,
                     custom_metric=es_method, # feval always gets raw prediction, 
                                          # custom_metric gets transformed prediction 
                                          # if not use custom objective
                     evals=pf_data,
                     evals_result=results,
                     xgb_model=bst_pf # continue training
                     )

    if not model_saved: # patience not reached
        print('Best iteration\tBest score\tBest number of trees:')
        print(f'{max_ind}\t{valid_scores[max_ind]}\t{max_ind+1}')
        print(f'Number of trees learned: {bst_nf.num_boosted_rounds()}')

        print(f'Slicing {max_ind+1} trees (best) from {bst_nf.num_boosted_rounds()} trees')
        bst_nf = bst_nf[:max_ind+1]
        bst_nf.save_model(model_file)
        print(f'Saved model {model_file}')

    results_bst_nf = {'dtrain': [(es_name, [train_scores[max_ind]])], 
                      'dtest': [(es_name, [test_scores[max_ind]])],
                      'dvalid': [(es_name, [valid_scores[max_ind]])]}
    if thred_range:
        # accuracy
        results_accuracy = get_accuracy(nf_data, bst_nf, thred_range)
        for d in results_accuracy:
            results_bst_nf[d].append(('Accuracy', results_accuracy[d]))

        # precision
        results_precision = get_precision(nf_data, bst_nf, thred_range)
        for d in results_precision:
            results_bst_nf[d].append(('Precision', results_precision[d]))

        # recall
        results_recall = get_recall(nf_data, bst_nf, thred_range)
        for d in results_recall:
            results_bst_nf[d].append(('Recall', results_recall[d]))

        # f1
        results_f1 = get_f1(nf_data, bst_nf, thred_range)
        for d in results_f1.keys():
            results_bst_nf[d].append(('F1', results_f1[d]))
    return results_bst_nf

