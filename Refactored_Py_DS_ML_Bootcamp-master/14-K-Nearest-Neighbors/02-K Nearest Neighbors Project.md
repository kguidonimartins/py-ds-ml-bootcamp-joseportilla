
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___

# K Nearest Neighbors Project 

Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.
## Import Libraries
**Import pandas,seaborn, and the usual libraries.**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

## Get the Data

**Read the 'KNN_Project_Data csv file into a dataframe**


```python
df = pd.read_csv('KNN_Project_Data')
df.columns = map(str.lower, df.columns)
df.columns = df.columns.str.replace(" ", "_")
```

**Check the head of the dataframe.**


```python
df.head()
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
      <th>xvpm</th>
      <th>gwyh</th>
      <th>trat</th>
      <th>tllz</th>
      <th>igga</th>
      <th>hykr</th>
      <th>edfs</th>
      <th>guub</th>
      <th>mgjm</th>
      <th>jhzc</th>
      <th>target_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1636.670614</td>
      <td>817.988525</td>
      <td>2565.995189</td>
      <td>358.347163</td>
      <td>550.417491</td>
      <td>1618.870897</td>
      <td>2147.641254</td>
      <td>330.727893</td>
      <td>1494.878631</td>
      <td>845.136088</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1013.402760</td>
      <td>577.587332</td>
      <td>2644.141273</td>
      <td>280.428203</td>
      <td>1161.873391</td>
      <td>2084.107872</td>
      <td>853.404981</td>
      <td>447.157619</td>
      <td>1193.032521</td>
      <td>861.081809</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1300.035501</td>
      <td>820.518697</td>
      <td>2025.854469</td>
      <td>525.562292</td>
      <td>922.206261</td>
      <td>2552.355407</td>
      <td>818.676686</td>
      <td>845.491492</td>
      <td>1968.367513</td>
      <td>1647.186291</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1059.347542</td>
      <td>1066.866418</td>
      <td>612.000041</td>
      <td>480.827789</td>
      <td>419.467495</td>
      <td>685.666983</td>
      <td>852.867810</td>
      <td>341.664784</td>
      <td>1154.391368</td>
      <td>1450.935357</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1018.340526</td>
      <td>1313.679056</td>
      <td>950.622661</td>
      <td>724.742174</td>
      <td>843.065903</td>
      <td>1370.554164</td>
      <td>905.469453</td>
      <td>658.118202</td>
      <td>539.459350</td>
      <td>1899.850792</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# EDA

Since this data is artificial, we'll just do a large pairplot with seaborn.

**Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**


```python
sns.pairplot(df, hue='target_class')
```

    /home/karlo/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    /home/karlo/anaconda3/lib/python3.7/site-packages/statsmodels/nonparametric/kde.py:488: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    /home/karlo/anaconda3/lib/python3.7/site-packages/statsmodels/nonparametric/kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    /home/karlo/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)





    <seaborn.axisgrid.PairGrid at 0x7f97ca1ea7b8>




![png](02-K%20Nearest%20Neighbors%20Project_files/02-K%20Nearest%20Neighbors%20Project_9_2.png)



```python

```

# Standardize the Variables

Time to standardize the variables.

**Import StandardScaler from Scikit learn.**


```python

```

**Create a StandardScaler() object called scaler.**


```python
scaler = StandardScaler()
```


```python

```

**Fit scaler to the features.**


```python
scaler.fit(df.drop('target_class', axis=1))
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```python

```

**Use the .transform() method to transform the features to a scaled version.**


```python
scaled_features = scaler.transform(df.drop('target_class', axis=1))
scaled_features
```




    array([[ 1.56852168, -0.44343461,  1.61980773, ..., -0.93279392,
             1.00831307, -1.06962723],
           [-0.11237594, -1.05657361,  1.7419175 , ..., -0.46186435,
             0.25832069, -1.04154625],
           [ 0.66064691, -0.43698145,  0.77579285, ...,  1.14929806,
             2.1847836 ,  0.34281129],
           ...,
           [-0.35889496, -0.97901454,  0.83771499, ..., -1.51472604,
            -0.27512225,  0.86428656],
           [ 0.27507999, -0.99239881,  0.0303711 , ..., -0.03623294,
             0.43668516, -0.21245586],
           [ 0.62589594,  0.79510909,  1.12180047, ..., -1.25156478,
            -0.60352946, -0.87985868]])




```python

```

**Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**


```python
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()
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
      <th>xvpm</th>
      <th>gwyh</th>
      <th>trat</th>
      <th>tllz</th>
      <th>igga</th>
      <th>hykr</th>
      <th>edfs</th>
      <th>guub</th>
      <th>mgjm</th>
      <th>jhzc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.568522</td>
      <td>-0.443435</td>
      <td>1.619808</td>
      <td>-0.958255</td>
      <td>-1.128481</td>
      <td>0.138336</td>
      <td>0.980493</td>
      <td>-0.932794</td>
      <td>1.008313</td>
      <td>-1.069627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.112376</td>
      <td>-1.056574</td>
      <td>1.741918</td>
      <td>-1.504220</td>
      <td>0.640009</td>
      <td>1.081552</td>
      <td>-1.182663</td>
      <td>-0.461864</td>
      <td>0.258321</td>
      <td>-1.041546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.660647</td>
      <td>-0.436981</td>
      <td>0.775793</td>
      <td>0.213394</td>
      <td>-0.053171</td>
      <td>2.030872</td>
      <td>-1.240707</td>
      <td>1.149298</td>
      <td>2.184784</td>
      <td>0.342811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011533</td>
      <td>0.191324</td>
      <td>-1.433473</td>
      <td>-0.100053</td>
      <td>-1.507223</td>
      <td>-1.753632</td>
      <td>-1.183561</td>
      <td>-0.888557</td>
      <td>0.162310</td>
      <td>-0.002793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.099059</td>
      <td>0.820815</td>
      <td>-0.904346</td>
      <td>1.609015</td>
      <td>-0.282065</td>
      <td>-0.365099</td>
      <td>-1.095644</td>
      <td>0.391419</td>
      <td>-1.365603</td>
      <td>0.787762</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check mean
df_feat.mean().round()
```




    xvpm    0.0
    gwyh    0.0
    trat    0.0
    tllz    0.0
    igga   -0.0
    hykr   -0.0
    edfs   -0.0
    guub   -0.0
    mgjm   -0.0
    jhzc    0.0
    dtype: float64




```python
# check sd
df_feat.std().round()
```




    xvpm    1.0
    gwyh    1.0
    trat    1.0
    tllz    1.0
    igga    1.0
    hykr    1.0
    edfs    1.0
    guub    1.0
    mgjm    1.0
    jhzc    1.0
    dtype: float64




```python

```

# Train Test Split

**Use train_test_split to split your data into a training set and a testing set.**


```python
X = df_feat
y = df['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
```


```python

```


```python

```

# Using KNN

**Import KNeighborsClassifier from scikit learn.**


```python

```

**Create a KNN model instance with n_neighbors=1**


```python
knn = KNeighborsClassifier(n_neighbors=1)
```


```python

```

**Fit this KNN model to the training data.**


```python
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=1, p=2,
               weights='uniform')




```python

```

# Predictions and Evaluations
Let's evaluate our KNN model!

**Use the predict method to predict values using your KNN model and X_test.**


```python
pred = knn.predict(X_test)
```


```python

```

**Create a confusion matrix and classification report.**


```python
print(confusion_matrix(y_test, pred))
```

    [[105  58]
     [ 35 102]]



```python

```


```python

```


```python
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.64      0.69       163
               1       0.64      0.74      0.69       137
    
       micro avg       0.69      0.69      0.69       300
       macro avg       0.69      0.69      0.69       300
    weighted avg       0.70      0.69      0.69       300
    



```python

```

# Choosing a K Value
Let's go ahead and use the elbow method to pick a good K Value!

**Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**


```python
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))    
```


```python

```

**Now create the following plot using the information from your for loop.**


```python
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error rate')
```




    Text(0, 0.5, 'Error rate')




![png](02-K%20Nearest%20Neighbors%20Project_files/02-K%20Nearest%20Neighbors%20Project_53_1.png)



```python

```

## Retrain with new K Value

**Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**


```python
knn = KNeighborsClassifier(n_neighbors=26)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

    [[133  30]
     [ 18 119]]
                  precision    recall  f1-score   support
    
               0       0.88      0.82      0.85       163
               1       0.80      0.87      0.83       137
    
       micro avg       0.84      0.84      0.84       300
       macro avg       0.84      0.84      0.84       300
    weighted avg       0.84      0.84      0.84       300
    



```python

```

# Great Job!
