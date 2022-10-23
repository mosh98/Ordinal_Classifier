
## Ordinal Classification/Regression

Turn your favourite classifier into an ordinal classifier.

Ordinal classification packages where every sklearn classifier is supported.

You need to have more than 2 labels for your features obviously!

The classification is based upon this paper over here:

[1] https://dl.acm.org/doi/10.1007/3-540-44795-4_13


```python
!git clone https://github.com/mosh98/Ordinal_Classifier.git
```

wait! i havent deployed to pypi yet, will do it when i feel like it.


### How do i use it?

1. import your favourite classifier from sklearn
2. Insert that clf into the OrdinalClassifier
3. Call fit and predict on your data

Here is an example: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CkoeHze9ee2WFe7P0sgqZS4Wyx9_rD8b?usp=sharing)

```python
from Ordinal_Classifier import Ordinal_Classifier as OC
from sklearn.tree import DecisionTreeClassifier

clf = OC.OrdinalClassifier(DecisionTreeClassifier())

clf.fit(X_train, y_train)
  
predictions = clf.predict(X_test)

# TADA! You just became a regression master! 
# By exploting an unemployed engineer. Remember that when you sleep 
```

#### Please pull an issue if you find any bugs

### Enjoy!!


