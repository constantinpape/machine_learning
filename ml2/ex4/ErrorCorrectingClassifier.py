import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ErrorCorrectingClassifier():

    def __init__(self, C, P, random_code):
        assert( P in (9,10,11) )
        self.C = C
        self.P = P
        self.random_code = random_code
        self.forests = None
        self.code_matrix = None

    # get random code matrix
    def _calculate_random_code_matrix(self):
        code_mat = np.random.binomial(1, 0.5, (self.C,self.P) )
        code_mat[np.where(code_mat == 0)] = -1
        assert( code_mat.shape == (self.C, self.P) )
        return code_mat

    # get code matrix
    def _calculate_correlation_code_matrix(self, data, target):
        # calculate normalized mean vectors
        mean_vectors = []
        for c in range(self.C):
            data_c = data[np.where(target == c)]
            vec = np.sum(data_c, axis = 0)
            norm = np.linalg.norm(vec)
            if norm != 0:
                vec /= norm
            else:
                "Mean vector of class", c, "has zero norm!"
            mean_vectors.append(vec)
        # calculate the covariance matrix
        cov_mat = np.zeros( (self.C, self.C) )
        for c1 in range(self.C):
            for c2 in range(self.C):
                cov_mat[c1,c2] = mean_vectors[c1].dot(mean_vectors[c2])
        # caclulate the eigenvalues of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov_mat)
        eigvecs = eigvecs.astype(np.float32)
        # take elementwise signum of the eigenvectors
        eigvecs = np.sign(eigvecs)
        # take care of 0 values, which have sign = 0
        eigvecs[np.where(eigvecs == 0)] = 1
        code_mat = np.zeros( (self.C,self.P) )
        # P = 10: use eigenvectors as codematrix
        if self.P == 10:
            assert( code_mat.shape == eigvecs.shape)
            code_mat = eigvecs
        # P = 9: remove eigenvector with the smallest eigenvalue
        elif self.P == 9:
            min_val = np.argmin(eigvals)
            eigvecs = np.delete(eigvecs, min_val, axis = 1)
            assert( code_mat.shape == eigvecs.shape)
            code_mat = eigvecs
        # P = 10: add arbitrary bit vector (with same number of 1 and -1)
        elif self.P == 11:
            vec_add = np.array( [1,1,1,1,1,-1,-1,-1,-1,-1] )
            np.random.shuffle( vec_add )
            vec_add = np.expand_dims(vec_add, axis = 1)
            eigvecs = np.concatenate( [eigvecs, vec_add], axis = 1)
            assert( code_mat.shape == eigvecs.shape)
            code_mat = eigvecs
        assert( code_mat.shape == (self.C, self.P) )
        return code_mat

    def get_code_matrix(self):
        if self.code_matrix == None:
            raise RuntimeError("Code Matrix not calculated yet!")
        else:
            return self.code_matrix

    def calculate_code_matrix(self, data, target):
        if self.random_code:
            code_mat = self._calculate_random_code_matrix()
        else:
            code_mat = self._calculate_correlation_code_matrix(data, target)
        return code_mat

    def fit(self, x_train, y_train):
        # get the code matrix
        self.code_matrix = self.calculate_code_matrix(x_train, y_train)
        self.forests = []
        for p in range(self.P):
            code = self.code_matrix[:,p]
            assert( code.shape[0] == self.C )
            target = np.zeros( x_train.shape[0] )
            for c in range(self.C):
                assert( code[c] == -1 or code[c] == 1)
                target[np.where(y_train == c)] = code[c]
            rfc = RandomForestClassifier(n_estimators = 20)
            rfc.fit(x_train, target)
            self.forests.append(rfc)

    def predict(self, x_test):
        prediction = np.zeros( x_test.shape[0] )
        for i in range( x_test.shape[0] ):
            x = x_test[i]
            new_code = np.zeros( self.P )
            for p in range(self.P):
                new_code[p] = self.forests[p].predict(x)
            # predict the class whose codeword has the smallest hamming distance to new_code
            from scipy.spatial.distance import hamming
            min_dist = float(np.inf)
            c_predicted = -1
            for c in range(self.C):
                dist = hamming(new_code, self.code_matrix[c,:])
                if dist < min_dist:
                    min_dist = dist
                    c_predicted = c
            prediction[i] = c_predicted
        return prediction
