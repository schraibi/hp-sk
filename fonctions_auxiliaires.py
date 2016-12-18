import re
import time
from sklearn.metrics import confusion_matrix
from hpsklearn import demo_support, components
import matplotlib.pyplot as plt

list_classifier = [ 'svc', 'knn', 'random_forest', 'extra_trees', 'ada_boost', 'gradient_boosting', 'sgd']
list_preprocessor =['pca', 'standard_scaler', 'min_max_scaler', 'normalizer', 'None']


def time_retriever(estimator):
    classifier_values = estimator.trials.trials
    timestamp = {}
    
    for i in range(len(classifier_values)):
        book_time = classifier_values[i]['book_time']
        timestamp['i'] = time.mktime(book_time.timetuple())
  
    return timestamp

####    

def predict_intermediate(classifiers, X_test):
    estim = estimator.hyperopt_estimator(classifier = classfiers)

    for i in range(classifiers.shape[0]):
        clf = classifiers[i][0]
        param = np.array(classifiers[i][1])
        estim = estimator.hyperopt_estimator(clf = clf(param),trial_timeout=60)
        y[i] = estim.predict(X_test)

    return(y)

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
    ax2 = fig.add_subplot(222)
    return demo_support.plot_minvalid_vs_time(estimator, ax2)
  