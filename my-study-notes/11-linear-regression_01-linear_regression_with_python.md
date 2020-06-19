# 11-linear-regression_01-linear_regression_with_python

Load libraries.

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import metrics

```

Import data.

```python

df = pd.read_csv('../Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv')

```

Check data.

```python
df.head()
df.info()
df.describe()
df.columns
```

Simple plots.

```python
sns.pairplot(df)
plt.show()

sns.distplot(df['Price'], kde=False)
sns.heatmap(data=df.corr(), cmap='viridis', annot=True)
```

Split data.

```python
# show columns names
df.columns

# create X df
X = df[
    [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms',
    'Area Population'
    ]
]

# create y df
y = df['Price']

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=101
    )

# Check data
X_train.shape
X_test .shape
y_train .shape
y_test .shape
```

Create a train model

```python
lm = LinearRegression()

# there is no need to create a new object for the lm.fit
lm.fit(X_train, y_train)
```

Check the model coefs.

```python
# check the intercept model
print(lm.intercept_)

# check the coefs for each feature (predictor)
lm.coef_

# organize into a dataframe
X_train.columns
coef_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
coef_df
```

> Explanation about the coefficients: Take the first one (`Avg. Area Income`), if we hold all other features fixed, a one unit increase of `Avg. Area Income` is associated with a increase of $21.52 in `Price`.

<!--
TODO: Test the same model in Boston dataset
 -->
Test the same model in Boston dataset

```python
boston = load_boston()
boston.keys()
print(boston['DESCR'])
print(boston['data'])
print(boston['feature_names'])
print(boston['target'])
boston.head()
```

## Get model predictions

```python
# get the predicted prices of the houses
predictions = lm.predict(X_test)
predictions

# get the real prices of the houses
y_test
```

> How far off are the `predictions` from the real prices (`y_test`)?

Check with a scatter plot!

```python
plt.scatter(y_test, predictions)
```

Check the residuals: the difference between the actual values (`y_test`) and the predicted values (`predictions`).

```python
sns.distplot((y_test - predictions))
```

> If the residuals are normally distributed, the model was a good choice for the data.


## Regression Evaluation Metrics

Here are three common evaluation metrics for regression problems:

> **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors.

> **Mean Squared Error** (MSE) is the mean of the squared errors.

> **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors.

Comparing these metrics:

- **MAE** is the easiest to understand, because it's the average error.
- **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
- **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.

All of these are **loss functions**, because we want to minimize them.

```python
print('MAE:', round(metrics.mean_absolute_error(y_test, predictions), 2))
print('MSE:', round(metrics.mean_squared_error(y_test, predictions), 2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, predictions)), 2))
```
