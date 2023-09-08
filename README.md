Code files
----------

- main.py: initial code file to learn model on one fold of data.
- find_best_parameters.py: search best hyperparameters setting of alpha, thredsholds for models.
- eval_fairness_metrics_rate.py: compute scores of fairness metrics of different models.
- eval_performance.py: compute performance scores of learned models on different metrics.

#### Execute code file by
`python find_best_parameters.py` or `python eval_fairness_metrics_rate.py`


Dependency
----------

- Python 3.8.13
- NumPy 1.23.3
- pandas 1.4.4
- scikit-learn 1.1.2
- SciPy 1.9.2
- XGBoost 1.6.2


Data
----

Due to the size limit, 10 folds cross-validation data can be downloaded from https://www.dropbox.com/s/rj4g5r4nt816guq/data.zip?dl=0

