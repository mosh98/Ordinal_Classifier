
## Ordinal Classification/Regression

Turn your favourite classifier into an ordinal classifier.

Ordinal classification packages where every sklearn classifier is supported.

You need to have more than 2 labels for your features obviously!

The classification is based upon this paper over here:

[1] https://dl.acm.org/doi/10.1007/3-540-44795-4_13


```python
pip install ordinal-classification
```

wait! i havent deployed to pypi yet, gimme a few days.


### How do i use it?

1. import your favourite classifier from sklearn
2. Insert that clf into the OrdinalClassifier
3. Call fit and predict on your data
4. Profit!
5. Profit!
6. repeat step 4 and 5 for even more profit!

```python
from oridnal_classifier import Ordinal_Classifier as OC
from sklearn import Decision Tree Classifier

clf = OC.OrdinalClassifier(DecisionTreeClassifier())

clf.fit(X_train, y_train)
  
predictions = clf.predict(X_test)

# TADA! You just became a regression master! 
# By exploting an unemployed engineer. Remember that when you sleep 
```


