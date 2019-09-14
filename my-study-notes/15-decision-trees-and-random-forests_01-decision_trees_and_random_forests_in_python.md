# 15-decision-trees-and-random-forests_01-decision_trees_and_random_forests_in_python

NOTE: See chapter 8 of ISLR book.

Load libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
```

Load data.

```python
df = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/kyphosis.csv')

df.head()

df.columns = map(str.lower, df.columns)

df.head()
```

Simple EDA

```python
sns.pairplot(df, hue='kyphosis')
```

Split data

```python
X = df.drop('kyphosis', axis=1)
y = df['kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

Model: Decision trees

```python
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
```

Evaluate

```python
dtree_predictions = dtree.predict(X_test)
print(confusion_matrix(y_test, dtree_predictions))
print(classification_report(y_test, dtree_predictions))
```

Model: Random Forest

```python
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
```

Evaluate

```python
rfc_predictions = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_predictions))
print(classification_report(y_test, rfc_predictions))
```
