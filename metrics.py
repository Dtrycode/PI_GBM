import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from scipy import interpolate, integrate
from typing import Tuple, List
from common import predictions, parameters
from utils import (
        sigmoid, 
        transfer_to_label, 
        compute_roc, 
        interpolate_roc_fun,
        slice_plot
        )
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

# Customized Metric Function
def nll(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    '''Mean negative log-likelihood metric.'''
    y = dtrain.get_label()
    #print('max: {}; min: {}'.format(max(predt), min(predt)))
    elements = sigmoid(predt)
    elements[y == 0] = 1 - elements[y == 0]
    #elements[elements == 0] = 1e-6 # for log
    #print('min: {}'.format(min(elements)))
    elements = np.log(elements)
    return 'PyNLL', -float(np.sum(elements) / len(y))

# Customized Metric Function for negative log-likelihood loss and adjusted KL divergence
def nll_np(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    '''Mean negative log-likelihood and adjusted KL divergence metric.'''
    y = dtrain.get_label()
    p = sigmoid(predt)

    elements = p.copy()
    elements[y == 0] = 1 - elements[y == 0]
    elements = np.log(elements)

    pD = predictions['pf_model']
    return 'PyNLL_NP', \
           -float(np.sum(elements - parameters['alpha'] \
           * (pD * (np.log(pD) - np.log(p)) + (1 - pD) * (np.log(1 - pD) - np.log(1 - p)))  \
           ) / len(y))

# Customized Metric Function for accuracy evaluation
def acc(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    '''Accuracy score'''
    return 'ACC', accuracy_score(dtrain.get_label(),
                                 transfer_to_label(sigmoid(predt), thred=parameters['thred']))

# Customized Metric Function for AUC PR evaluation
def aucpr(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    '''AUC Precision Recall score'''
    return 'AUCPR', average_precision_score(dtrain.get_label(), sigmoid(predt))

# Customized Metric Function for AUC ROC evaluation
def aucroc(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    '''AUC ROC score'''
    return 'AUCROC', roc_auc_score(dtrain.get_label(), sigmoid(predt))

# Wrap score function to a loss function
def make_loss(func):
    def wrap(*args, **kwargs):
        key, value = func(*args, **kwargs)
        return key, -value
    return wrap

# Fairness metric: delta equal opportunity
def delta_eo(predt: np.ndarray, X: np.ndarray, y: np.ndarray, 
             protect_group: List[int]) -> Tuple[str, float]:
    '''Delta Equal Opportunity'''
    predt = predt[y == 1]
    X = X[y == 1]
    return 'EO', [abs(predt[X[:, i] == protect_group[i]].mean() - \
                      predt[X[:, i] != protect_group[i]].mean()) \
                  for i in range(X.shape[1])]

# Fairness metric: delta demographic parity
def delta_dp(predt: np.ndarray, X: np.ndarray, 
             protect_group: List[int]) -> Tuple[str, float]:
    '''Delta Demographic Parity'''
    return 'DP', [abs(predt[X[:, i] == protect_group[i]].mean() - \
                      predt[X[:, i] != protect_group[i]].mean()) \
                  for i in range(X.shape[1])]

# Fairness metric: Equalized Odds
def calculate_eo(X, y, pred_classes, protect_group):
    label_pos = y == 1
    label_neg = ~ label_pos

    pred_pos = pred_classes == 1
    pred_neg = ~ pred_pos

    tp_loc = label_pos & pred_pos
    tn_loc = label_neg & pred_neg
    fp_loc = label_neg & pred_pos
    fn_loc = label_pos & pred_neg

    outputs = []

    for i in range(X.shape[1]):
        prot_g = X[:, i] == protect_group[i]
        non_prot_g = ~ prot_g
        # protected group
        protected_pos = (prot_g & pred_pos).sum()
        protected_neg = (prot_g & pred_neg).sum()

        tp_protected = (prot_g & tp_loc).sum()
        tn_protected = (prot_g & tn_loc).sum()

        fn_protected = (prot_g & fn_loc).sum()
        fp_protected = (prot_g & fp_loc).sum()
        # non protected group
        non_protected_pos = (non_prot_g & pred_pos).sum()
        non_protected_neg = (non_prot_g & pred_neg).sum()

        tp_non_protected = (non_prot_g & tp_loc).sum()
        tn_non_protected = (non_prot_g & tn_loc).sum()

        fn_non_protected = (non_prot_g & fn_loc).sum()
        fp_non_protected = (non_prot_g & fp_loc).sum()
        # fairness
        tpr_protected = tp_protected / (tp_protected + fn_protected)
        tnr_protected = tn_protected / (tn_protected + fp_protected)

        tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
        tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

        C_prot = protected_pos / (protected_pos + protected_neg)
        C_non_prot = non_protected_pos / (non_protected_pos + non_protected_neg)

        stat_par = C_non_prot - C_prot

        output = {}

        output['balanced_accuracy'] = 0.5 * \
            ((tp_protected + tp_non_protected) / \
             (tp_protected + tp_non_protected + fn_protected + fn_non_protected) + \
            (tn_protected + tn_non_protected) / \
             (tn_protected + tn_non_protected + fp_protected + fp_non_protected))

        output['Equalized Odds'] = abs(tpr_non_protected - tpr_protected) + \
                             abs(tnr_non_protected - tnr_protected)

        output["tpr_protected"] = tpr_protected
        output["tpr_non_protected"] = tpr_non_protected
        output["tnr_protected"] = tnr_protected
        output["tnr_non_protected"] = tnr_non_protected

        outputs.append(output)

    return outputs


# Fairness metric: Statistical Parity
def calculate_sp(X, y, pred_classes, protect_group):
    label_pos = y == 1
    label_neg = ~ label_pos

    pred_pos = pred_classes == 1
    pred_neg = ~ pred_pos

    tp_loc = label_pos & pred_pos
    tn_loc = label_neg & pred_neg
    fp_loc = label_neg & pred_pos
    fn_loc = label_pos & pred_neg

    outputs = []

    for i in range(X.shape[1]):
        prot_g = X[:, i] == protect_group[i]
        non_prot_g = ~ prot_g
        # protected group
        protected_pos = (prot_g & pred_pos).sum()
        protected_neg = (prot_g & pred_neg).sum()

        tp_protected = (prot_g & tp_loc).sum()
        tn_protected = (prot_g & tn_loc).sum()

        fn_protected = (prot_g & fn_loc).sum()
        fp_protected = (prot_g & fp_loc).sum()
        # non protected group
        non_protected_pos = (non_prot_g & pred_pos).sum()
        non_protected_neg = (non_prot_g & pred_neg).sum()

        tp_non_protected = (non_prot_g & tp_loc).sum()
        tn_non_protected = (non_prot_g & tn_loc).sum()

        fn_non_protected = (non_prot_g & fn_loc).sum()
        fp_non_protected = (non_prot_g & fp_loc).sum()
        # fairness
        tpr_protected = tp_protected / (tp_protected + fn_protected)
        tnr_protected = tn_protected / (tn_protected + fp_protected)

        tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
        tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

        C_prot = protected_pos / (protected_pos + protected_neg)
        C_non_prot = non_protected_pos / (non_protected_pos + non_protected_neg)

        stat_par = C_non_prot - C_prot

        output = {}

        output['balanced_accuracy'] = 0.5 * \
            ((tp_protected + tp_non_protected) / \
             (tp_protected + tp_non_protected + fn_protected + fn_non_protected) + \
            (tn_protected + tn_non_protected) / \
             (tn_protected + tn_non_protected + fp_protected + fp_non_protected))

        output['Statistical Parity'] = stat_par

        output['Positive_prot_pred'] = C_prot
        output['Positive_non_prot_pred'] = C_non_prot
        output['Negative_prot_pred'] = protected_neg / (protected_pos + protected_neg)
        output['Negative_non_prot_pred'] = non_protected_neg / \
            (non_protected_pos + non_protected_neg)

        outputs.append(output)

    return outputs


# Fairness metric: Absolute Between-ROC Area
def calculate_abroca(X, y, preds, protect_group, 
        n_grid=10000, plot_slices=False, lb=0, ub=1, limit=1000,
        filename='dataset.abroca.pdf'):
    '''
    X - numpy array of privileged features
    y - true labels (should be 0,1 only)
    preds - predicted probabilities
    protect_group - protected group value for each sensitive feature
    n_grid (optional) - number of grid points to use in approximation (numeric) (default of 10000 is more than adequate for most cases)
    plot_slices (optional) - if TRUE, ROC slice plots are generated and saved to file_name (boolean)
    lb (optional) - Lower limit of integration (use -numpy.inf for -infinity) Default is 0
    ub (optional) - Upper limit of integration (use -numpy.inf for -infinity) Default is 1
    limit (optional) - An upper bound on the number of subintervals used in the adaptive algorithm.Default is 1000
    filename (optional) - File name (including directory) to save the slice plot generated

    Returns Abroca value
    '''
    if not ((preds >= 0) & (preds <= 1)).all():
        print("predictions must be in range [0,1]")
        exit(1)
    if len(np.unique(y)) != 2:
        print("The label column should be binary")
        exit(1)

    sls = []
    for i in range(X.shape[1]):
        fpr_tpr_dict = {}
        prot_loc = X[:, i] == protect_group[i]
        nonprot_loc = ~ prot_loc
        fpr_tpr_dict['prot'] = compute_roc(preds[prot_loc], y[prot_loc])
        fpr_tpr_dict['nonprot'] = compute_roc(preds[nonprot_loc], 
                                              y[nonprot_loc])
        # compare protected to non-protected class 
        # accumulate absolute difference btw ROC curves to slicing statistic
        nonprot_roc_x, nonprot_roc_y = interpolate_roc_fun(
            fpr_tpr_dict['nonprot'][0],
            fpr_tpr_dict['nonprot'][1],
            n_grid,
        )
        prot_roc_x, prot_roc_y = interpolate_roc_fun(
            fpr_tpr_dict['prot'][0],
            fpr_tpr_dict['prot'][1],
            n_grid,
        )

        # use function approximation to compute slice statistic via 
        # piecewise linear function
        if list(nonprot_roc_x) == list(prot_roc_x):
            f1 = interpolate.interp1d(x=nonprot_roc_x, 
                    y=(nonprot_roc_y - prot_roc_y))
            f2 = lambda x, a: abs(f1(x))
            sl, _ = integrate.quad(f2, lb, ub, limit)
        else:
            print("Non-Prot and Prot FPR are different")
            exit(1)

        if plot_slices == True:
            fns = filename.split('.')
            fni = f'{".".join(fns[:-1])}{i}.{fns[-1]}'
            slice_plot(
                nonprot_roc_x,
                prot_roc_x,
                nonprot_roc_y,
                prot_roc_y,
                nonprot_group_name='Non-Protected',
                prot_group_name='Protected',
                fout=fni,
                value=round(sl, 4),
            )

        sls.append(sl)

    return sls
