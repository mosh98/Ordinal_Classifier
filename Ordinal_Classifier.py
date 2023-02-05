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
        self.uniques_class = None

    def fit(self,X,y):
        self.uniques_class = np.sort(np.unique(y))
        assert self.uniques_class.shape[0] >= 3, f'OrdinalClassifier needs at least 3 classes, only {self.uniques_class.shape[0]} found'

        for i in range(self.uniques_class.shape[0]-1):
            binary_y = (y > self.uniques_class[i]).astype(np.uint8)
            
            clf = clone(self.clf)
            clf.fit(X,binary_y)
            self.clfs[i] = clf

    def predict(self,X):
        return np.argmax( self.predict_proba(X), axis=1 )

    def predict_proba(self,X):
        predicted = [self.clfs[k].predict_proba(X)[:,1].reshape(-1,1) for k in self.clfs]

        p_x_first = 1-predicted[0]
        p_x_last  = predicted[-1]
        p_x_middle= [predicted[i] - predicted[i+1] for i in range(len(predicted) - 1)]
        
        probs = np.hstack([p_x_first, *p_x_middle, p_x_last])

        return probs

    def set_params(self, **params):
        self.clf.set_params(**params)
        for _,clf in self.clfs.items():
            clf.set_params(**params)


