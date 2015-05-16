import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn

class OneVsRestClassifier():

    def __init__( self, C, use_subsample, predict_probabilities ):
        self.C = C
        self.use_subsample = use_subsample
        self.predict_probabilities = predict_probabilities
        self.forests = None

    def fit(self, x_train, y_train):
        self.forests = []
        # train the random forests
        for c in range(self.C):
            y_train_c = -1 * np.ones( y_train.shape )
            y_train_c[ np.where( y_train == c ) ] = 1
            # if use_subsample is true set the class weights
            if self.use_subsample:
                # TODO make this work, need current sklearn!
                rfc = RandomForestClassifier( class_weight = "subsample" )
            else:
                rfc = RandomForestClassifier()
            rfc.fit(x_train, y_train_c)
            self.forests.append(rfc)

    def predict(self, x_test):
        if self.predict_probabilities:
            return self._predict_probabilities(x_test)
        else:
            return self._predict_predictions(x_test)

    def _predict_probabilities(self, x_test):
        prediction = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            x = x_test[i]
            probabilities = np.zeros( self.C )
            for c in range( self.C ):
                probabilities[c] = self.forests[c].predict_proba(x)[0][1]
            prediction[i] = np.argmax( probabilities )
        return prediction

    def  _predict_predictions(self, x_test):
        prediction = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            x = x_test[i]
            predictions_sample = np.zeros( self.C )
            for c in range( self.C ):
                predictions_sample[c] = self.forests[c].predict(x)
            preds_pos = np.where(predictions_sample == 1)[0]
            # if no class was predicted by a classifier or more than one classifier predicted a class, return -1 -> dont know
            if not preds_pos.shape[0] == 1:
                prediction[i] = -1
            else:
                prediction[i] = preds_pos[0]
        return prediction
