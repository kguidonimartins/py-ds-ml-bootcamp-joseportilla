# 16-support-vector-machines_01-support_vector_machines_with_python

NOTE: See chapter 9 of ISLR book.

> A non-probabilistic binary linear classifier. Support vector classifier and logistic regression are closely related.

Since there is a limited number of the line to separate data in a hyperplane, SVM algorithm choose a hyperplane that maximizes the margin between classes.

For linear non-separable classes, "kernel trick" create a third dimension.

Parameters:
- C: the amount that the margin can be violated by the n observations. If C = 0 then there is no violations to the margin, simply amounts to the maximal margin hyperplane (of course, a maximal margin hyperplane exists only if the two classes are separable). C controls the bias-variance trade-off of the statistical learning technique. When **C is small**, we seek narrow margins that are rarely violated; this amounts to a classifier that is highly fit to the data, which may have **low bias but high variance**. On the other hand, when **C is larger**, the margin is wider and we allow more violations to it; this amounts to fitting the data less hard and obtaining a classifier that is potentially **more biased but may have lower variance**.
- kernel: enlarge the feature space in order to accommodate a non-linear boundary between the classes.


Load libraries

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

```

Load data

```python

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()

# dataset description
print(cancer['DESCR'])

# get faetures
df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df_feat.head()

# get target
cancer['target']
cancer['target_names']

```

Simple EDA

```python

sns.pairplot(df_feat)

```

Split data

```python

X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

```

Train data

```python

svm = SVC()
svm.fit(X_train, y_train)

```

Predict

```python

pred_svm = svm.predict(X_test)

```

Evaluate

```python

print(confusion_matrix(y_test, pred_svm))
print(classification_report(y_test, pred_svm))

```

Using `GridSearchCV`

```python

param_grid = {
	'C': [0.1, 1, 10, 100, 1000],
	'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}

grid = GridSearchCV(SVC(), param_grid, verbose=3, n_jobs=-1)

grid.fit(X_train, y_train)

grid.best_params_

grid.cv_results_

```

Get predictions from grid best parameters

```python

pred_grid = grid.predict(X_test)

```

Evaluate

```python

print(confusion_matrix(y_test, pred_grid))

print(classification_report(y_test, pred_grid))

```
