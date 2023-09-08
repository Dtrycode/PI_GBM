import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from utils import print_accuracy, sigmoid
from metrics import acc, aucpr, aucroc, make_loss
from objectives import neg_logl, neg_logl_np
from common import models, parameters, predictions
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

def get_early_stopping_method(early_stopping_method):
    ''' Choose early stopping metric from auc-pr, auc-roc, accuracy
    '''
    if early_stopping_method == 'aucpr':
        es_method = aucpr
        es_name = 'AUCPR'
    elif early_stopping_method == 'aucroc':
        es_method = aucroc
        es_name = 'AUCROC'
    elif early_stopping_method == 'acc':
        es_method = acc
        es_name = 'ACC'
    else:
        raise NotImplementedError(f'{early_stopping_method} is not implemented.')
    return es_name, es_method


def load_and_check_model(path_train, path_test, model_file, feature_types=None):
    '''
    Arguments:
      path_train: path of train data
      path_test: path of test data
      model_file: file path to save trained model
    '''
    if feature_types:
        dtrain = xgb.DMatrix(path_train, feature_types=feature_types, enable_categorical=True)
        dtest = xgb.DMatrix(path_test, feature_types=feature_types, enable_categorical=True)
    else:
        dtrain = xgb.DMatrix(path_train)
        dtest = xgb.DMatrix(path_test)

    bst = xgb.Booster()
    
    # load trained model
    bst.load_model(model_file)
    print(f'Loaded model {model_file}')

    # ‘gain’ - the average gain across all splits the feature is used in
    print('Feature importance (gain): {}'.format(bst.get_score(importance_type='gain')))
    # ‘weight’ - the number of times a feature is used to split the data across all trees
    print('Feature importance (weight): {}'.format(bst.get_score(importance_type='weight')))
    
    # print accuracy
    print_accuracy([[dtrain, 'train'], [dtest, 'test']], bst, [0.7511, 0.5])

    # print AUC PR
    print(f'AUC PR of train: {average_precision_score(dtrain.get_label(), sigmoid(bst.predict(dtrain)))}')
    print(f'AUC PR of test: {average_precision_score(dtest.get_label(), sigmoid(bst.predict(dtest)))}')
    # print AUC ROC
    print(f'AUC ROC of train: {roc_auc_score(dtrain.get_label(), sigmoid(bst.predict(dtrain)))}')
    print(f'AUC ROC of test: {roc_auc_score(dtest.get_label(), sigmoid(bst.predict(dtest)))}')
    return bst


# functions to train, save and load model
def train_and_save_model(path_train, 
                         path_test, 
                         model_file, 
                         path_valid=None, 
                         num_tree=5,
                         early_stopping_rounds=1,
                         early_stopping_method='aucpr',
                         max_depth=6,
                         learning_rate=0.3,
                         max_cat_to_onehot=5,
                         feature_types=None):
    '''
    Arguments:
      path_train: path of train data
      path_test: path of test data
      model_file: file path to save trained model
    '''
    es_name, es_method = get_early_stopping_method(early_stopping_method)

    if feature_types:
        dtrain = xgb.DMatrix(path_train, feature_types=feature_types, enable_categorical=True)
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
        dtest = xgb.DMatrix(path_test)
        tree_method='exact'
        tree_parameters = {'tree_method': tree_method, 'seed': 1994,
                           'disable_default_eval_metric': 1,
                           'max_depth': max_depth, # maximum tree depth
                           'eta': learning_rate # learning rate
                          }

    if path_valid:
        if feature_types:
            dvalid = xgb.DMatrix(path_valid, 
                                 feature_types=feature_types, 
                                 enable_categorical=True)
        else:
            dvalid = xgb.DMatrix(path_valid)

        results = {}
        bst = xgb.train(tree_parameters,
                        dtrain=dtrain,
                        num_boost_round=num_tree,
                        obj=neg_logl,
                        custom_metric=make_loss(es_method), # feval always gets raw prediction, 
                                     # custom_metric gets transformed prediction 
                                     # if not use custom objective
                        evals=[(dtrain, 'dtrain'), (dtest, 'dtest'), (dvalid, 'dvalid')],
                        early_stopping_rounds=early_stopping_rounds, 
                                                 # works on loss function, not score function
                                                 # loss decreases
                        evals_result=results)

        # find the turning point from validation
        #turning_iteration, turning_score = find_turning_point(results['dvalid'])
        print('Best iteration\tBest score\tBest number of trees:')
        print(f'{bst.best_iteration}\t{bst.best_score}\t{bst.best_ntree_limit}') 
        print(f'Number of trees learned: {bst.num_boosted_rounds()}')
        # save trained model
        #bst.save_model(f'{model_file[:-5]}_{turning_iteration}.json')
        print(f'Slicing {bst.best_iteration+1} trees (best) from {bst.num_boosted_rounds()} trees')
        bst = bst[:bst.best_iteration+1]
        bst.save_model(model_file)
        print(f'Saved model {model_file}')
    else:
        results = {}
        bst = xgb.train(tree_parameters,
                        dtrain=dtrain,
                        num_boost_round=num_tree,
                        obj=neg_logl,
                        custom_metric=es_method, # feval always gets raw prediction, 
                                     # custom_metric gets transformed prediction 
                                     # if not use custom objective
                        evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
                        evals_result=results)
    
        # save trained model
        bst.save_model(model_file)
        print(f'Saved model {model_file}')

    # print accuracy
    if path_valid:
        data = [[dtrain, 'train'], [dvalid, 'valid'], [dtest, 'test']]
    else:
        data = [[dtrain, 'train'], [dtest, 'test']]
    print_accuracy(data, bst, [0.7511, 0.5])


def train_and_save_model_with_pf(path_train, 
                                 path_test, 
                                 model_file, 
                                 path_valid=None, 
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

    es_name, es_method = get_early_stopping_method(early_stopping_method)

    if feature_types:
        dtrain = xgb.DMatrix(path_train, feature_types=feature_types, enable_categorical=True)
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
        dtest = xgb.DMatrix(path_test)
        tree_method='exact'
        tree_parameters = {'tree_method': tree_method, 'seed': 1994,
                           'disable_default_eval_metric': 1,
                           'max_depth': max_depth, # maximum tree depth
                           'eta': learning_rate # learning rate
                          }

    if path_valid:
        if feature_types:
            dvalid = xgb.DMatrix(path_valid,
                                 feature_types=feature_types,
                                 enable_categorical=True)
        else:
            dvalid = xgb.DMatrix(path_valid)

        results = {}
        bst = xgb.train(tree_parameters,
                        dtrain=dtrain,
                        num_boost_round=num_tree,
                        obj=neg_logl_np,
                        custom_metric=make_loss(es_method), # feval always gets raw prediction, 
                                             # custom_metric gets transformed prediction 
                                             # if not use custom objective
                        evals=[(dtrain, 'dtrain'), (dtest, 'dtest'), (dvalid, 'dvalid')],
                        early_stopping_rounds=early_stopping_rounds, 
                                                 # works on loss function, not score function
                                                 # loss decreases
                        evals_result=results)

        print('Best iteration\tBest score\tBest number of trees:')
        print(f'{bst.best_iteration}\t{bst.best_score}\t{bst.best_ntree_limit}')
        print(f'Number of trees learned: {bst.num_boosted_rounds()}')
    
        # save trained model
        print(f'Slicing {bst.best_iteration+1} trees (best) from {bst.num_boosted_rounds()} trees')
        bst = bst[:bst.best_iteration+1]
        bst.save_model(model_file)
        print(f'Saved model {model_file}')

    else:
        results = {}
        bst = xgb.train(tree_parameters,
                        dtrain=dtrain,
                        num_boost_round=num_tree,
                        obj=neg_logl_np,
                        custom_metric=es_method, # feval always gets raw prediction, 
                                             # custom_metric gets transformed prediction 
                                             # if not use custom objective
                        evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
                        evals_result=results)
    
        # save trained model
        bst.save_model(model_file)
        print(f'Saved model {model_file}')

    # print accuracy
    if path_valid:
        data = [[dtrain, 'train'], [dvalid, 'valid'], [dtest, 'test']]
    else:
        data = [[dtrain, 'train'], [dtest, 'test']]
    print_accuracy(data, bst, [0.7511, 0.5])


def train_and_save_model_nf_pf(path_nf_train, path_nf_test, 
                               path_pf_train, path_pf_test, 
                               model_file, 
                               path_nf_valid=None, path_pf_valid=None,
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

    es_name, es_method = get_early_stopping_method(early_stopping_method)

    if feature_types_nf:
        dtrain_nf = xgb.DMatrix(path_nf_train, 
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
        dtest_pf = xgb.DMatrix(path_pf_test)
        tree_method_pf = 'exact'
        tree_parameters_pf = {'tree_method': tree_method_pf, 'seed': 1994,
                              'disable_default_eval_metric': 1,
                              'max_depth': max_depth, # maximum tree depth
                              'eta': learning_rate # learning rate
                             }

    use_valid = path_nf_valid is not None
    model_saved = False

    if use_valid:
        if feature_types_nf:
            dvalid_nf = xgb.DMatrix(path_nf_valid,
                                    feature_types=feature_types_nf,
                                    enable_categorical=True)
        else:
            dvalid_nf = xgb.DMatrix(path_nf_valid)
        if feature_types_pf:
            dvalid_pf = xgb.DMatrix(path_pf_valid,
                                    feature_types=feature_types_pf,
                                    enable_categorical=True)
        else:
            dvalid_pf = xgb.DMatrix(path_pf_valid)
        valid_scores = []
        nf_data = [(dtrain_nf, 'dtrain'), (dtest_nf, 'dtest'), (dvalid_nf, 'dvalid')]
        pf_data = [(dtrain_pf, 'dtrain'), (dtest_pf, 'dtest'), (dvalid_pf, 'dvalid')]
    else:
        nf_data = [(dtrain_nf, 'dtrain'), (dtest_nf, 'dtest')]
        pf_data = [(dtrain_pf, 'dtrain'), (dtest_pf, 'dtest')]


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
        # if validation and early stopping is applied
        if use_valid:
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

    if not model_saved:
        if use_valid: # patience not reached
            print('Best iteration\tBest score\tBest number of trees:')
            print(f'{max_ind}\t{valid_scores[max_ind]}\t{max_ind+1}')
            print(f'Number of trees learned: {bst_nf.num_boosted_rounds()}')

            print(f'Slicing {max_ind+1} trees (best) from {bst_nf.num_boosted_rounds()} trees')
            bst_nf = bst_nf[:max_ind+1]
        bst_nf.save_model(model_file)
        print(f'Saved model {model_file}')

    # print accuracy
    if path_nf_valid:
        data_nf = [[dtrain_nf, 'train'], [dvalid_nf, 'valid'], [dtest_nf, 'test']]
    else:
        data_nf = [[dtrain_nf, 'train'], [dtest_nf, 'test']]
    print_accuracy(data_nf, bst_nf, [0.7511, 0.5])
