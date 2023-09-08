import numpy as np
import xgboost as xgb
from typing import Tuple
from common import parameters, predictions
from utils import sigmoid
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

# Customized Objective Function for negative log-likelihood loss
def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient for negative log-likelihood.'''
    y = dtrain.get_label()
    return -(y - predt)

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for negative log-likelihood.'''
    return predt * (1 - predt)

def neg_logl(predt: np.ndarray,
             dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Negative Log-likelihood objective. Transform from log-likelihood by
       adding negative sign.
    '''
    predt = sigmoid(predt)
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

# Customized Objective Function for negative log-likelihood loss and adjusted KL divergence
def gradient_np(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient for negative log-likelihood.'''
    y = dtrain.get_label()
    return -(y - predt - parameters['alpha'] * (predt - predictions['pf_model']))

def hessian_np(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for negative log-likelihood.'''
    return (1 + parameters['alpha']) * predt * (1 - predt)

def neg_logl_np(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Negative Log-likelihood objective. Transform from log-likelihood by
       adding negative sign.
    '''
    predt = sigmoid(predt)
    grad = gradient_np(predt, dtrain)
    hess = hessian_np(predt, dtrain)
    return grad, hess
