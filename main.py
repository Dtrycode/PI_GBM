from all_models import (
        load_and_check_model,
        train_and_save_model, 
        train_and_save_model_with_pf, 
        train_and_save_model_nf_pf
    )
from common import parameters, models, predictions
from utils import get_prediction

# for now, only work on binary classification problem
if __name__ == '__main__':
    parameters['thred'] = 0.7511
    parameters['alpha'] = 0.15

    model_name = 'pi' # 'normal', 'pi', 'pi*'
    kf = 9 # index of cross-validation fold
    
    num_tree = 20
    early_stopping_rounds = 5
    early_stopping_method = 'aucroc'
    dataset = 'adult'

    path_nf_train = f'data/{dataset}/fold{kf}/{dataset}.txt.nf.train'
    path_nf_valid = f'data/{dataset}/fold{kf}/{dataset}.txt.nf.valid'
    path_nf_test = f'data/{dataset}/fold{kf}/{dataset}.txt.nf.test'
    feature_types_nf = ['q']*4+['c']*6

    path_pf_train = f'data/{dataset}/fold{kf}/{dataset}.txt.pf.train'
    path_pf_valid = f'data/{dataset}/fold{kf}/{dataset}.txt.pf.valid'
    path_pf_test = f'data/{dataset}/fold{kf}/{dataset}.txt.pf.test'
    feature_types_pf = ['q']+['c']*2

    if model_name == 'normal':
        path_nf_model = f'saved_models/{dataset}_nf_bst{kf}.json'
        # train and save a model with normal features
        train_and_save_model(path_nf_train,
                             path_nf_test,
                             path_nf_model,
                             path_valid=path_nf_valid,
                             num_tree=num_tree,
                             early_stopping_rounds=early_stopping_rounds,
                             early_stopping_method=early_stopping_method,
                             feature_types=feature_types_nf)

    elif model_name == 'pi':
        path_pf_model = f'saved_models/{dataset}_pf_bst{kf}.json'
        path_np_model = f'saved_models/{dataset}_np_bst{kf}.json'

        # train a boosting model with only privileged features
        train_and_save_model(path_pf_train,
                             path_pf_test,
                             path_pf_model,
                             path_valid=path_pf_valid,
                             num_tree=num_tree,
                             early_stopping_rounds=early_stopping_rounds,
                             early_stopping_method=early_stopping_method,
                             feature_types=feature_types_pf)

        # then use that model to provide more fine-grained probability distribution
        # for each instance when train model with normal features
        # check whether the performance is good
        # provide custom objective and metric again after the model is loaded
        models['pf_model'] = load_and_check_model(path_pf_train,
                                                  path_pf_test,
                                                  path_pf_model,
                                                  feature_types=feature_types_pf)
        predictions['pf_model'] = get_prediction(path_pf_train,
                                                 models['pf_model'],
                                                 feature_types=feature_types_pf)

        train_and_save_model_with_pf(path_nf_train, 
                                     path_nf_test, 
                                     path_np_model, 
                                     path_valid=path_nf_valid, 
                                     num_tree=num_tree,
                                     early_stopping_rounds=early_stopping_rounds,
                                     early_stopping_method=early_stopping_method,
                                     feature_types=feature_types_nf)

    elif model_name == 'pi*':
        path_nf_pf_model = f'saved_models/{dataset}_nf_pf_bst{kf}.json'
        # Co-ordinate gradient descent training of nf and pf model
        train_and_save_model_nf_pf(path_nf_train=path_nf_train, 
                                   path_nf_test=path_nf_test, 
                                   path_pf_train=path_pf_train, 
                                   path_pf_test=path_pf_test, 
                                   model_file=path_nf_pf_model, 
                                   path_nf_valid=path_nf_valid, 
                                   path_pf_valid=path_pf_valid,
                                   num_tree=num_tree,
                                   early_stopping_rounds=early_stopping_rounds,
                                   early_stopping_method=early_stopping_method,
                                   feature_types_nf=feature_types_nf,
                                   feature_types_pf=feature_types_pf,
                                   verbose=True)
