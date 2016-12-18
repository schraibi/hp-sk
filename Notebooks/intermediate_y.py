from __future__ import print_function
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from hyperopt import tpe
import hpsklearn
import sys

def test_abalone(X_train, y_train, X_test, y_test):

     estimator = hpsklearn.HyperoptEstimator(
        preprocessing=hpsklearn.components.any_preprocessing('pp'),
        classifier=hpsklearn.components.any_classifier('clf'),
        algo=tpe.suggest,
        trial_timeout=15.0,  # seconds
        max_evals=10,
        seed=1
    )

    # /BEGIN `Demo version of estimator.fit()`
    print('', file=sys.stderr)
    print('====Demo classification on Iris dataset====', file=sys.stderr)

    iterator = estimator.fit_iter(X_train, y_train)
    next(iterator)
    y =[]
    y[0] = estimator.predict(X_test)
    n_trial = 0
    while len(estimator.trials.trials) < estimator.max_evals:
        iterator.send(1)  # -- try one more model
        y[n_trial -1] = estimator.predict(X_test)
        n_trial += 1
        print('Trial', n_trial, 'loss:', estimator.trials.losses()[-1],'y:' 
              file=sys.stderr)
        # hpsklearn.demo_support.scatter_error_vs_time(estimator)
        # hpsklearn.demo_support.bar_classifier_choice(estimator)

    estimator.retrain_best_model_on_full_data(X_train, y_train)

    # /END Demo version of `estimator.fit()`

    print('Test accuracy:', estimator.score(X_test, y_test), file=sys.stderr)
    print('====End of demo====', file=sys.stderr)
    return y
