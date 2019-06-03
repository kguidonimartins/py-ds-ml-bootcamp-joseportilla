
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___

# Tensorflow with Estimators

As we saw previously how to build a full Multi-Layer Perceptron model with full Sessions in Tensorflow. Unfortunately this was an extremely involved process. However developers have created Estimators that have an easier to use flow!

It is much easier to use, but you sacrifice some level of customization of your model. Let's go ahead and explore it!

## Get the Data

We will the iris data set.

Let's get the data:


```python
import pandas as pd
```


```python
df = pd.read_csv('iris.csv')
```


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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']
```


```python
X = df.drop('target',axis=1)
y = df['target'].apply(int)
```

## Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

# Estimators

Let's show you how to use the simpler Estimator interface!


```python
import tensorflow as tf
```

    C:\Users\Marcial\Anaconda3\lib\site-packages\h5py\__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


## Feature Columns


```python
X.columns
```




    Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype='object')




```python
feat_cols = []

for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
```


```python
feat_cols
```




    [_NumericColumn(key='sepal_length', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     _NumericColumn(key='sepal_width', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     _NumericColumn(key='petal_length', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     _NumericColumn(key='petal_width', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]



## Input Function


```python
# there is also a pandas_input_fn we'll see in the exercise!!
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)
```


```python
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3,feature_columns=feat_cols)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\Marcial\AppData\Local\Temp\tmp3_l8l99d
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\Marcial\\AppData\\Local\\Temp\\tmp3_l8l99d', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x0000020ED5FA2390>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
classifier.train(input_fn=input_func,steps=50)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into C:\Users\Marcial\AppData\Local\Temp\tmp3_l8l99d\model.ckpt.
    INFO:tensorflow:loss = 15.285385, step = 1
    INFO:tensorflow:Saving checkpoints for 50 into C:\Users\Marcial\AppData\Local\Temp\tmp3_l8l99d\model.ckpt.
    INFO:tensorflow:Loss for final step: 3.4342575.





    <tensorflow.python.estimator.canned.dnn.DNNClassifier at 0x20edb48f748>



## Model Evaluation

** Use the predict method from the classifier model to create predictions from X_test **


```python
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
```


```python
note_predictions = list(classifier.predict(input_fn=pred_fn))
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\Marcial\AppData\Local\Temp\tmp3_l8l99d\model.ckpt-50
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.



```python
note_predictions[0]
```




    {'class_ids': array([2], dtype=int64),
     'classes': array([b'2'], dtype=object),
     'logits': array([-3.6269774 ,  0.16824062,  1.2134217 ], dtype=float32),
     'probabilities': array([0.00581369, 0.2586391 , 0.7355472 ], dtype=float32)}




```python
final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])
```

** Now create a classification report and a Confusion Matrix. Does anything stand out to you?**


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test,final_preds))
```

    [[20  0  0]
     [ 0  6  0]
     [ 0  0 19]]



```python
print(classification_report(y_test,final_preds))
```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00        20
              1       1.00      1.00      1.00         6
              2       1.00      1.00      1.00        19
    
    avg / total       1.00      1.00      1.00        45
    


# Great Job!
