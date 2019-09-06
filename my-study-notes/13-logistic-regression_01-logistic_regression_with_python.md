# 13-logistic-regression_01-logistic_regression_with_python

See sections 4-4.3 of ISLR book.

> Logistic Regression as a method for classification.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```

Read data.

<!-- TODO: Do more feature engineering for this dataset.  -->

```python
train = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv')
train.head()
```

Visualizing missing data.

```python
train.isnull()

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
```

Some exploratory data analysis.

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', data=train, hue='Sex')
sns.countplot(x='Survived', data=train, hue='Pclass')
sns.distplot(train['Age'].dropna())
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(bins=100, figsize=(10, 4))
```

Imputation.

```python
plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=train)
```

Using median of `Age` per `Pclass` to impute the missing values of `Age`.

```python
train.groupby('Pclass').median()['Age']

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
```

Remove 'Cabin' column.

```python
train.drop('Cabin', axis=1, inplace=True)
train.head()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
```

Removing the remaining missing values regardless of the column that occurs.

```python
train.dropna(inplace=True)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
```

Dealing with categorical variables.

```python
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embark], axis=1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()
train.drop('PassengerId', axis=1, inplace=True)
train.head()
```

## Modeling

Splitting the data

```python
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

Creating a model instance

```python
logmodel = LogisticRegression()
```

Fitting the model

```python
logmodel.fit(X_train, y_train)
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='warn',
#           n_jobs=None, penalty='l2', random_state=None, solver='warn',
#           tol=0.0001, verbose=0, warm_start=False)
```

Predicting

```python
predictions = logmodel.predict(X_test)
```

Validating

NOTE: About `metrics.classification_report`.
See Gueron (2019, pág. 92), Hands-On Machine Learning with Scikit-Learn and TensorFlow (2ª Edition)
- **Precision**: accuracy of the positive predictions (TP / TP + FP)
- **Recall**: ratio of positive instances that are correctly detected by the classifier (TP / TP + FN)
- **F1**: the *harmonic mean* of precision and recall. Whereas the *regular mean* treats all values equally, the *harmonic mean* gives much more weight to low values. As result, the classifier will only get a high **F1** if both recall and precision are high.


```python
print(classification_report(y_test, predictions))
confusion_matrix(y_test, predictions)
```
