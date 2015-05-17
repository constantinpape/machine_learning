import numpy as np
from sklearn.ensemble import RandomForestClassifier

class OneVsOneClassifier():

    def __init__(self, C, greedy):
        self.C = C
        self.greedy = greedy
        self.forests = None
        self.index_to_pair = self.make_index_to_pair(C)
        self.N_pairs = C * (C - 1) / 2

    # TODO make this work for C != 10
    def make_index_to_pair(self, C):
        index_to_pair = {  0 : (0,1),  1 : (0,2),  2 : (0,3),  3 : (0,4),  4 : (0,5),
                           5 : (0,6),  6 : (0,7),  7 : (0,8),  8 : (0,9),  9 : (1,2),
                          10 : (1,3), 11 : (1,4), 12 : (1,5), 13 : (1,6), 14 : (1,7),
                          15 : (1,8), 16 : (1,9), 17 : (2,3), 18 : (2,4), 19 : (2,5),
                          20 : (2,6), 21 : (2,7), 22 : (2,8), 23 : (2,9), 24 : (3,4),
                          25 : (3,5), 26 : (3,6), 27 : (3,7), 28 : (3,8), 29 : (3,9),
                          30 : (4,5), 31 : (4,6), 32 : (4,7), 33 : (4,8), 34 : (4,9),
                          35 : (5,6), 36 : (5,7), 37 : (5,8), 38 : (5,9), 39 : (6,7),
                          40 : (6,8), 41 : (6,9), 42 : (7,8), 43 : (7,9), 44 : (8,9)}
        return index_to_pair

    def fit(self, x_train, y_train):
        if self.greedy:
            self._fit_greedy(x_train, y_train)
        else:
            self._fit_pairwise(x_train, y_train)

    def _fit_pairwise(self, x_train, y_train):
        self.forests = []
        # train a classifier on each pair of classes
        for index in range(self.N_pairs):
            c0 = self.index_to_pair[index][0]
            c1 = self.index_to_pair[index][1]
            mask0 = np.where(y_train == c0)
            mask1 = np.where(y_train == c1)
            rfc = RandomForestClassifier(n_estimators = 20)
            train_data = np.concatenate(
                    [ x_train[mask0],
                      x_train[mask1] ] )
            target     = np.concatenate(
                    [ np.ones(   mask0[0].shape[0] ),
                     -1*np.ones( mask1[0].shape[0] ) ] )
            assert( target.shape[0] == train_data.shape[0] )
            rfc.fit(train_data, target)
            self.forests.append(rfc)

    def _fit_greedy(self, x_train, y_train):
        self.forests = []
        # train 9 classifiers, each on 2 consecutive classes
        for index in range( self.C - 1 ):
            rfc = RandomForestClassifier(n_estimators = 20)
            c0 = index
            c1 = index + 1
            mask0 = np.where( y_train == c0 )
            mask1 = np.where( y_train == c1 )
            train_data = np.concatenate(
                    [ x_train[mask0],
                      x_train[mask1] ] )
            target     = np.concatenate(
                    [ -1*np.ones( mask0[0].shape[0] ),
                      np.ones(    mask1[0].shape[0] ) ] )
            assert( train_data.shape[0] == target.shape[0] )
            rfc.fit(train_data, target)
            self.forests.append(rfc)

    def predict(self, x_test):
        if self.greedy:
            return self._predict_greedy(x_test)
        else:
            return self._predict_pairwise(x_test)

    # predict the class with most votes
    def _predict_pairwise(self, x_test):
        prediction = np.zeros( x_test.shape[0] )
        for i in range( x_test.shape[0] ):
            x = x_test[i]
            count_predictions = np.zeros( self.C )
            for index in range(self.N_pairs):
                if self.forests[index].predict(x)[0] == 1.:
                    c = self.index_to_pair[index][0]
                else:
                    c = self.index_to_pair[index][1]
                count_predictions[c] += 1
            prediction[i] = np.argmax( count_predictions )
        return prediction

    def _predict_greedy(self, x_test):
        prediction = np.zeros( x_test.shape[0] )
        for i in range( x_test.shape[0] ):
            x = x_test[i]
            a = 0
            b = self.C - 1
            while a != b:
                # this probabilities (0,1) yields the best result in cross validation
                prob_a = self.forests[a].predict_proba(x)[0][0]
                prob_b = self.forests[b-1].predict_proba(x)[0][1]
                #print prob_a, prob_b
                if prob_a > prob_b:
                    b -= 1
                else:
                    a += 1
            prediction[i] = a
        return prediction
