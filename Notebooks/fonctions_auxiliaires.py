import re
import datetime  
import time
from sklearn.metrics import confusion_matrix

def retriever(estimator):
    classifier_values = estimator.trials.trials
    timestamp = {}
    classifier_param = {}
    
    for i in range(len(classifier_values)):
        time['i'] = classifier_values[i]['book_time']
        timestamp['i'] = time.mktime(time['i'].timetuple())
        iteration[''] = classifier_values[i]['misc']['vals']


####    

def predict_intermediate(classifiers, X_test):
    estim = estimator.hyperopt_estimator(classifier = classfiers

    for i in range(classifiers.shape[0]):
        clf = classifiers[i][0]
        param = np.array(classifiers[i][1])
        estim = estimator.hyperopt_estimator(clf = clf(param),trial_timeout=60)
        y[i] = estim.predict(X_test)

    return(y)

####

def metrics(estimator, X_train, y_train, X_test, y_test, y)
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y)
    precision = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1]
    recall = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[1,0]
    F = 2*(precison*recall)/(precision + recall)
    
    return train_score, test_score, precison, recall, F


####

