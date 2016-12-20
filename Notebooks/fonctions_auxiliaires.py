from __future__ import print_function
import time
from sklearn.metrics import confusion_matrix
from hpsklearn import demo_support, components
import matplotlib.pyplot as plt
import sys

list_classifier = [ 'svc', 'knn', 'random_forest', 'extra_trees', 'ada_boost', 'gradient_boosting', 'sgd']
list_preprocessor =['pca', 'standard_scaler', 'min_max_scaler', 'normalizer', 'None']


def time_retriever(estimator):
    classifier_values = estimator.trials.trials
    timestamp = {}
    
    for i in range(len(classifier_values)):
        book_time = classifier_values[i]['book_time']
        timestamp[i] = time.mktime(book_time.timetuple())
  
    return timestamp

####    

def fit_intermediate(estimator, X_train, y_train, X_test, y_test):
   
    predictions = []
    accuracies = []
    iterator = estimator.fit_iter(X_train, y_train)
    next(iterator)

    n_trial = 0
    while len(estimator.trials.trials) < estimator.max_evals:
        iterator.send(1)  # -- try one more model
        n_trial += 1
        print('Trial', n_trial, 'loss:', estimator.trials.losses()[-1], file=sys.stderr)
        # hpsklearn.demo_support.scatter_error_vs_time(estimator)
        # hpsklearn.demo_support.bar_classifier_choice(estimator)
        estimator.retrain_best_model_on_full_data(X_train, y_train)
        predictions.append(estimator.predict(X_test))
        accuracies.append(estimator.score(X_test, y_test))
    # /END Demo version of `estimator.fit()`

    print('Test accuracy:', estimator.score(X_test, y_test), file=sys.stderr)
    print('Predict:', estimator.predict(X_test), file=sys.stderr)
    print('Best Model:', estimator.best_model(), file=sys.stderr)
    print('====End of demo====', file=sys.stderr)
    return predictions, accuracies

####

def metrics(estimator, X_train, y_train, X_test, y_test, y):
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y)
    precision = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[1,0])
    F = 2*(precision*recall)/(precision + recall)
    
    return train_score, test_score, precision, recall, F


####

def plot_perf(estimator):
    
    fig = plt.figure()

    ax= fig.add_subplot(111)
    demo_support.scatter_error_vs_time(estimator, ax)
    ax2 = fig.add_subplot(111)
    demo_support.plot_minvalid_vs_time(estimator, ax2)
  
