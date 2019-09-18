
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Support Vector Machines Project 

Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!

## The Data
For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

Here's a picture of the three different Iris types:


```python
# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)
```




![jpeg](02-Support%20Vector%20Machines%20Project_files/02-Support%20Vector%20Machines%20Project_1_0.jpeg)




```python
# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)
```




![jpeg](02-Support%20Vector%20Machines%20Project_files/02-Support%20Vector%20Machines%20Project_2_0.jpeg)




```python
# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)
```




![jpeg](02-Support%20Vector%20Machines%20Project_files/02-Support%20Vector%20Machines%20Project_3_0.jpeg)



The iris dataset contains measurements for 150 iris flowers from three different species.

The three classes in the Iris dataset:

    Iris-setosa (n=50)
    Iris-versicolor (n=50)
    Iris-virginica (n=50)

The four features of the Iris dataset:

    sepal length in cm
    sepal width in cm
    petal length in cm
    petal width in cm

## Get the data

**Use seaborn to get the iris data by using: iris = sns.load_dataset('iris')**


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

iris = sns.load_dataset('iris')
```


```python
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize the data and get you started!

## Exploratory Data Analysis

Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!

**Import some libraries you think you'll need.**


```python

```

**Create a pairplot of the data set. Which flower species seems to be the most separable?**


```python
sns.pairplot(iris, hue='species', markers=["s", "o", "D"], palette='viridis')
```




    <seaborn.axisgrid.PairGrid at 0x7f27d7ca8eb8>




![png](02-Support%20Vector%20Machines%20Project_files/02-Support%20Vector%20Machines%20Project_10_1.png)



```python

```




    <seaborn.axisgrid.PairGrid at 0x12afb9cc0>




![png](02-Support%20Vector%20Machines%20Project_files/02-Support%20Vector%20Machines%20Project_11_1.png)


**Create a kde plot of sepal_length versus sepal width for setosa species of flower.**


```python
df = iris[iris['species'] == 'setosa']

sns.kdeplot(df['sepal_width'], df['sepal_length'], cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f27d6467400>




![png](02-Support%20Vector%20Machines%20Project_files/02-Support%20Vector%20Machines%20Project_13_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x12f102080>




![png](02-Support%20Vector%20Machines%20Project_files/02-Support%20Vector%20Machines%20Project_14_1.png)


# Train Test Split

**Split your data into a training set and a testing set.**


```python
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
```


```python

```


```python

```

# Train a Model

Now its time to train a Support Vector Machine Classifier. 

**Call the SVC() model from sklearn and fit the model to the training data.**


```python
svc_default = SVC()
svc_default.fit(X_train, y_train)
```

    /home/karlo/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)





    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python

```


```python

```


```python

```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



## Model Evaluation

**Now get predictions from the model and create a confusion matrix and a classification report.**


```python
svc_default_pred = svc_default.predict(X_test)
```


```python
print(confusion_matrix(y_test, svc_default_pred))
```

    [[13  0  0]
     [ 0 20  0]
     [ 0  0 12]]



```python
print(classification_report(y_test, svc_default_pred))
```

                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        13
      versicolor       1.00      1.00      1.00        20
       virginica       1.00      1.00      1.00        12
    
       micro avg       1.00      1.00      1.00        45
       macro avg       1.00      1.00      1.00        45
    weighted avg       1.00      1.00      1.00        45
    



```python

```


```python

```


```python

```

    [[15  0  0]
     [ 0 13  1]
     [ 0  0 16]]



```python

```

                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00        15
     versicolor       1.00      0.93      0.96        14
      virginica       0.94      1.00      0.97        16
    
    avg / total       0.98      0.98      0.98        45
    


Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

## Gridsearch Practice

** Import GridsearchCV from SciKit Learn.**


```python

```

**Create a dictionary called param_grid and fill out some parameters for C and gamma.**


```python
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}
```


```python

```

**Create a GridSearchCV object and fit it to the training data.**


```python
grid = GridSearchCV(SVC(), param_grid, verbose=3, n_jobs=-1, cv=5)
```


```python
grid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done 125 out of 125 | elapsed:    0.7s finished
    /home/karlo/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)





    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)




```python
grid.best_params_
```




    {'C': 1, 'gamma': 0.1}




```python

```

    Fitting 3 folds for each of 16 candidates, totalling 48 fits
    [CV] gamma=1, C=0.1 ..................................................
    [CV] ......................................... gamma=1, C=0.1 -   0.0s
    [CV] gamma=1, C=0.1 ..................................................
    [CV] ......................................... gamma=1, C=0.1 -   0.0s
    [CV] gamma=1, C=0.1 ..................................................
    [CV] ......................................... gamma=1, C=0.1 -   0.0s
    [CV] gamma=0.1, C=0.1 ................................................
    [CV] ....................................... gamma=0.1, C=0.1 -   0.0s
    [CV] gamma=0.1, C=0.1 ................................................
    [CV] ....................................... gamma=0.1, C=0.1 -   0.0s
    [CV] gamma=0.1, C=0.1 ................................................
    [CV] ....................................... gamma=0.1, C=0.1 -   0.0s
    [CV] gamma=0.01, C=0.1 ...............................................
    [CV] ...................................... gamma=0.01, C=0.1 -   0.0s
    [CV] gamma=0.01, C=0.1 ...............................................
    [CV] ...................................... gamma=0.01, C=0.1 -   0.0s
    [CV] gamma=0.01, C=0.1 ...............................................
    [CV] ...................................... gamma=0.01, C=0.1 -   0.0s
    [CV] gamma=0.001, C=0.1 ..............................................
    [CV] ..................................... gamma=0.001, C=0.1 -   0.0s
    [CV] gamma=0.001, C=0.1 ..............................................
    [CV] ..................................... gamma=0.001, C=0.1 -   0.0s
    [CV] gamma=0.001, C=0.1 ..............................................
    [CV] ..................................... gamma=0.001, C=0.1 -   0.0s
    [CV] gamma=1, C=1 ....................................................
    [CV] ........................................... gamma=1, C=1 -   0.0s
    [CV] gamma=1, C=1 ....................................................
    [CV] ........................................... gamma=1, C=1 -   0.0s
    [CV] gamma=1, C=1 ....................................................
    [CV] ........................................... gamma=1, C=1 -   0.0s
    [CV] gamma=0.1, C=1 ..................................................
    [CV] ......................................... gamma=0.1, C=1 -   0.0s
    [CV] gamma=0.1, C=1 ..................................................
    [CV] ......................................... gamma=0.1, C=1 -   0.0s
    [CV] gamma=0.1, C=1 ..................................................
    [CV] ......................................... gamma=0.1, C=1 -   0.0s
    [CV] gamma=0.01, C=1 .................................................
    [CV] ........................................ gamma=0.01, C=1 -   0.0s
    [CV] gamma=0.01, C=1 .................................................
    [CV] ........................................ gamma=0.01, C=1 -   0.0s
    [CV] gamma=0.01, C=1 .................................................
    [CV] ........................................ gamma=0.01, C=1 -   0.0s
    [CV] gamma=0.001, C=1 ................................................
    [CV] ....................................... gamma=0.001, C=1 -   0.0s
    [CV] gamma=0.001, C=1 ................................................
    [CV] ....................................... gamma=0.001, C=1 -   0.0s
    [CV] gamma=0.001, C=1 ................................................
    [CV] ....................................... gamma=0.001, C=1 -   0.0s
    [CV] gamma=1, C=10 ...................................................
    [CV] .......................................... gamma=1, C=10 -   0.0s
    [CV] gamma=1, C=10 ...................................................
    [CV] .......................................... gamma=1, C=10 -   0.0s
    [CV] gamma=1, C=10 ...................................................
    [CV] .......................................... gamma=1, C=10 -   0.0s
    [CV] gamma=0.1, C=10 .................................................
    [CV] ........................................ gamma=0.1, C=10 -   0.0s
    [CV] gamma=0.1, C=10 .................................................
    [CV] ........................................ gamma=0.1, C=10 -   0.0s
    [CV] gamma=0.1, C=10 .................................................
    [CV] ........................................ gamma=0.1, C=10 -   0.0s
    [CV] gamma=0.01, C=10 ................................................
    [CV] ....................................... gamma=0.01, C=10 -   0.0s
    [CV] gamma=0.01, C=10 ................................................
    [CV] ....................................... gamma=0.01, C=10 -   0.0s
    [CV] gamma=0.01, C=10 ................................................
    [CV] ....................................... gamma=0.01, C=10 -   0.0s
    [CV] gamma=0.001, C=10 ...............................................
    [CV] ...................................... gamma=0.001, C=10 -   0.0s
    [CV] gamma=0.001, C=10 ...............................................
    [CV] ...................................... gamma=0.001, C=10 -   0.0s
    [CV] gamma=0.001, C=10 ...............................................
    [CV] ...................................... gamma=0.001, C=10 -   0.0s
    [CV] gamma=1, C=100 ..................................................
    [CV] ......................................... gamma=1, C=100 -   0.0s
    [CV] gamma=1, C=100 ..................................................
    [CV] ......................................... gamma=1, C=100 -   0.0s
    [CV] gamma=1, C=100 ..................................................
    [CV] ......................................... gamma=1, C=100 -   0.0s
    [CV] gamma=0.1, C=100 ................................................
    [CV] ....................................... gamma=0.1, C=100 -   0.0s
    [CV] gamma=0.1, C=100 ................................................
    [CV] ....................................... gamma=0.1, C=100 -   0.0s
    [CV] gamma=0.1, C=100 ................................................
    [CV] ....................................... gamma=0.1, C=100 -   0.0s
    [CV] gamma=0.01, C=100 ...............................................
    [CV] ...................................... gamma=0.01, C=100 -   0.0s
    [CV] gamma=0.01, C=100 ...............................................
    [CV] ...................................... gamma=0.01, C=100 -   0.0s
    [CV] gamma=0.01, C=100 ...............................................
    [CV] ...................................... gamma=0.01, C=100 -   0.0s
    [CV] gamma=0.001, C=100 ..............................................
    [CV] ..................................... gamma=0.001, C=100 -   0.0s
    [CV] gamma=0.001, C=100 ..............................................
    [CV] ..................................... gamma=0.001, C=100 -   0.0s
    [CV] gamma=0.001, C=100 ..............................................
    [CV] ..................................... gamma=0.001, C=100 -   0.0s


    [Parallel(n_jobs=1)]: Done  40 tasks       | elapsed:    0.2s
    [Parallel(n_jobs=1)]: Done  48 out of  48 | elapsed:    0.2s finished





    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'gamma': [1, 0.1, 0.01, 0.001], 'C': [0.1, 1, 10, 100]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=2)



**Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**


```python
svc_grid_pred = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test, svc_grid_pred))
```

    [[13  0  0]
     [ 0 19  1]
     [ 0  0 12]]



```python
print(classification_report(y_test, svc_grid_pred))
```

                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        13
      versicolor       1.00      0.95      0.97        20
       virginica       0.92      1.00      0.96        12
    
       micro avg       0.98      0.98      0.98        45
       macro avg       0.97      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    



```python

```


```python

```

    [[15  0  0]
     [ 0 13  1]
     [ 0  0 16]]



```python

```

                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00        15
     versicolor       1.00      0.93      0.96        14
      virginica       0.94      1.00      0.97        16
    
    avg / total       0.98      0.98      0.98        45
    


You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

## Great Job!
