import numpy as np
import xgboost as xgb
from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score,
        roc_curve
        )
from scipy import interpolate
import matplotlib.pyplot as plt
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def transfer_to_label(x, thred=0.5):
    label_x = np.zeros(x.shape)
    label_x[x >= thred] = 1
    return label_x

def get_prediction(path_train, model, feature_types=None):
    ''' the prediction is on train data
    Arguments:
      path_train: path of train data
      model: xgboost booster model
    '''
    if feature_types:
        dtrain = xgb.DMatrix(path_train, feature_types=feature_types, enable_categorical=True)
    else:
        dtrain = xgb.DMatrix(path_train)
    bst = model
    return sigmoid(bst.predict(dtrain))

def print_accuracy(data, model, threds):
    for thred in threds:
        print(f'Print accuracy at thredhold {thred} ----------')
        for d in data:
            print(f'Accuracy of {d[1]}: {accuracy_score(d[0].get_label(), transfer_to_label(sigmoid(model.predict(d[0])), thred=thred))}')

def get_accuracy(data, model, threds):
    results = {}
    for d in data:
        results[d[1]] = [accuracy_score(d[0].get_label(), \
                transfer_to_label(sigmoid(model.predict(d[0])), thred=thred)) for thred in threds]
    return results

def get_precision(data, model, threds):
    results = {}
    for d in data:
        results[d[1]] = [precision_score(d[0].get_label(), \
                transfer_to_label(sigmoid(model.predict(d[0])), thred=thred)) for thred in threds]
    return results

def get_recall(data, model, threds):
    results = {}
    for d in data:
        results[d[1]] = [recall_score(d[0].get_label(), \
                transfer_to_label(sigmoid(model.predict(d[0])), thred=thred)) for thred in threds]
    return results

def get_f1(data, model, threds):
    results = {}
    for d in data:
        results[d[1]] = [f1_score(d[0].get_label(), \
                transfer_to_label(sigmoid(model.predict(d[0])), thred=thred)) for thred in threds]
    return results

def argmax_r(result):
    '''return the last occurrence of the max value'''
    return len(result) - np.argmax(result[::-1]) - 1

def compute_roc(y_scores, y_true):
    '''
    Function to compute the Receiver Operating Characteristic (ROC) curve for a set of predicted probabilities and the true class labels.
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns FPR and TPR values
    '''
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr, tpr

def interpolate_roc_fun(fpr, tpr, n_grid):
    '''
    Function to Use interpolation to make approximate the Receiver Operating Characteristic (ROC) curve along n_grid equally-spaced values.
    fpr - vector of false positive rates computed from compute_roc
    tpr - vector of true positive rates computed from compute_roc
    n_grid - number of approximation points to use (default value of 10000 more than adequate for most applications) (numeric)

    Returns  a list with components x and y, containing n coordinates which  interpolate the given data points according to the method (and rule) desired
    '''
    roc_approx = interpolate.interp1d(x=fpr, y=tpr)
    x_new = np.linspace(0, 1, num=n_grid)
    y_new = roc_approx(x_new)
    return x_new, y_new

def slice_plot(
    nonprot_roc_fpr,
    prot_roc_fpr,
    nonprot_roc_tpr,
    prot_roc_tpr,
    nonprot_group_name,
    prot_group_name,
    fout="./slice_plot.png",
    value=0.0
):
    '''
    Function to create a 'slice plot' of two roc curves with area between them (the ABROCA region) shaded.

    nonprot_roc_fpr, prot_roc_fpr - FPR of nonprot and prot groups
    nonprot_roc_tpr, prot_roc_tpr - TPR of nonprot and prot groups
    nonprot_group_name - (optional) - nonprot group display name on the slice plot
    prot_group_name - (optional) - prot group display name on the slice plot
    fout - (optional) -  File name (including directory) to save the slice plot generated

    No return value; displays slice plot & file is saved to disk
    '''
    plt.figure(1, figsize=(5, 4))
    title = 'ABROCA = ' + str(value)
    plt.title(title)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.ylim((-0.04, 1.04))
    plt.plot(
        nonprot_roc_fpr,
        nonprot_roc_tpr,
        label='{o}'.format(o=nonprot_group_name),
        linestyle='-',
        color='r',
    )
    plt.plot(
        prot_roc_fpr,
        prot_roc_tpr,
        label='{o}'.format(o=prot_group_name),
        linestyle='-',
        color='b',
    )
    plt.fill(
        nonprot_roc_fpr.tolist() + np.flipud(prot_roc_fpr).tolist(),
        nonprot_roc_tpr.tolist() + np.flipud(prot_roc_tpr).tolist(),
        'y',
    )
    plt.legend()
    plt.savefig(fout, bbox_inches = 'tight')
    #plt.show()

def mask_adult(X):
    # mask age, race to binary
    # age: 17.0, ..., 90.0 -> 25-60, <25 or >60
    # race: 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
    #       -> Non-White, White
    x0p = (X[:, 0] < 25) | (X[:, 0] > 60)
    x0n = ~ x0p
    X[x0p, 0] = 1
    X[x0n, 0] = 0

    x1p = X[:, 1] < 4
    x1n = ~ x1p
    X[x1p, 1] = 0
    X[x1n, 1] = 1

    return X


def mask_credit(X):
    # mask education, marriage
    # education: 1 = graduate school; 2 = university; 3 = high school; 4 = others
    #            (0, 1, 2, 3)
    #            -> 0 = not-university; 1 = university
    # marriage: 1 = married; 2 = single; 3 = others -> 0 = not-single; 1 = single
    #           (0, 1, 2)
    x1p = X[:, 1] != 1
    x1n = ~ x1p
    X[x1p, 1] = 0
    X[x1n, 1] = 1

    x2p = X[:, 2] != 1
    x2n = ~ x2p
    X[x2p, 2] = 0
    X[x2n, 2] = 1

    return X

def mask_numom2b_b(X):
    # keep the leftmost bit of Race1-8
    X = X[:, 0]
    X = X.reshape(X.shape[0], 1)

    return X

def get_tpr(labels, pred_classes):
    tp = ((labels == 1) & (pred_classes == 1)).sum()
    fn = ((labels == 1) & (pred_classes == 0)).sum()
    return tp / (tp + fn)
