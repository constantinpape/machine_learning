import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from OneVsRestClassifier import OneVsRestClassifier
from OneVsOneClassifier import OneVsOneClassifier
from ErrorCorrectingClassifier import ErrorCorrectingClassifier

# Number of classes
C = 10

# load digts dataset from sklearn
def load_data():
    from sklearn.datasets import load_digits
    digits = load_digits()
    data = digits["data"]
    images = digits["images"]
    target = digits["target"]
    target_names = digits["target_names"]
    # number of instances, classes and dimensions
    N = data.shape[0]
    # normalize the data
    # first substract the mean feature vector
    mean = np.mean(data, axis = 0)
    mean = np.tile(mean, (N,1) )
    data -= np.mean(data)
    # then divide by the standard deviation
    std = np.std(data, axis = 0)
    std = np.tile(std, (N,1) )
    data = np.divide(data, std)
    # seperate the data by classes
    data_return   = []
    target_return = []
    for c in target_names:
        data_c = data[np.where(target == c)]
        data_return.append(data_c)
        target_c = c * np.ones(data_c.shape[0])
        target_return.append(target_c)
    #data_return   = np.concatenate( [d for d in data_return], axis = 0  )
    #target_return = np.concatenate( [t for t in target_return], axis = 0)
    #print data_return.shape, target_return.shape
    return data_return, target_return

def evaluate(prediction, truth):
    assert( prediction.shape[0] == truth.shape[0] )
    correct = (prediction == truth)
    return float( np.where( correct == True )[0].shape[0] ) / prediction.shape[0]

# evaluate the classifier via cross validation
def evaluate_classifier(classifier, data, target):
    res = 0.
    N = 10
    # make the cross validation data
    from sklearn import cross_validation
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    for c in range(C):
        x_train_list_c = []
        y_train_list_c = []
        x_test_list_c = []
        y_test_list_c = []
        data_c = data[c]
        target_c = target[c]
        kf = cross_validation.KFold(target_c.shape[0], n_folds = N)
        for train_index, test_index in kf:
            x_train, x_test = data_c[train_index], data_c[test_index]
            y_train, y_test = target_c[train_index], target_c[test_index]
            x_train_list_c.append(x_train)
            y_train_list_c.append(y_train)
            x_test_list_c.append(x_test)
            y_test_list_c.append(y_test)
        x_train_list.append(x_train_list_c)
        y_train_list.append(y_train_list_c)
        x_test_list.append(x_test_list_c)
        y_test_list.append(y_test_list_c)
    for i in range(N):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for c in range(C):
            x_train.append(x_train_list[c][i])
            y_train.append(y_train_list[c][i])
            x_test.append(x_test_list[c][i])
            y_test.append(y_test_list[c][i])
        x_train = np.concatenate( [x for x in x_train ], axis = 0)
        y_train = np.concatenate( [y for y in y_train ], axis = 0)
        x_test = np.concatenate( [x for x in x_test ], axis = 0)
        y_test = np.concatenate( [y for y in y_test ], axis = 0)
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        res += evaluate(y_test, prediction)
    return res / N


def evaluate_one_vs_rest(data, target):
    for pred_probs in (False, True):
        #for use_subsample in (False, True):
        use_subsample = False
        classifier = OneVsRestClassifier( C, use_subsample, pred_probs )
        res = evaluate_classifier(classifier, data, target)
        print "Result for OneVsRest, predict_probabilities =", pred_probs, ", use_subsample =", use_subsample, ":", res

def evaluate_one_vs_one(data, target):
    for greedy in (False, True):
        classifier = OneVsOneClassifier( C, greedy )
        res = evaluate_classifier(classifier, data, target)
        print "Result for OneVsOne, greedy =", greedy, ":", res

def evaluate_error_correcting(data, target):
    for P in (9,10,11):
        for random_code in (False, True):
            classifier = ErrorCorrectingClassifier(C, P, random_code)
            res = evaluate_classifier(classifier, data, target)
            mat = classifier.get_code_matrix()
            dist = 0.5 * ( P - np.max( mat.dot(mat.transpose()) - P * np.eye(C) ) )
            print "Maximum Code distance, random_code =", random_code, ", P =", P, ":", dist
            print "Result for ErrorCorrecting, P =", P, ", random_code =", random_code, ":", res

def evaluate_multiclass_rf(data, target):
    # use 10 * number of trees in other classifiers (default value is 10)
    classifier = RandomForestClassifier(n_estimators = 100)
    res = evaluate_classifier(classifier, data, target)
    print "Result for multiclass random forest:", res

if __name__ == '__main__':
    data, target = load_data()
    #evaluate_one_vs_rest(data, target))
    #evaluate_one_vs_one(data, target)
    #evaluate_error_correcting(data, target)
    evaluate_multiclass_rf(data, target)
