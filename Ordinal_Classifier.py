from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    #https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
    """
    A classifier that can be trained on a range of classes.
    @param classifier: A scikit-learn classifier.
    """
    def __init__(self,clf):
        self.clf = clf
        self.clfs = {}


    def fit(self,X,y):
            self.uniques_class = np.sort(np.unique(y))
            if self.uniques_class.shape[0] > 2:
                for i in range(self.uniques_class.shape[0]-1):
                    #binary_y = (y > self.uniques_class[1]).astype(np.uint8)

                    binary_y = (y > self.uniques_class[i]).astype(np.uint8)
                    clf = clone(self.clf)
                    clf.fit(X,binary_y)
                    self.clfs[i] = clf

    def predict(self,X):
        return np.argmax( self.predict_proba(X), axis=1 )

    def predict_proba(self,X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}

        predicted = []

        for i,y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:
                #Vi = Pr(y>Vi-1)- Pr(y > Vi)
                predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                predicted.append(clfs_predict[y-1][:,1])

        return np.vstack(predicted).T


