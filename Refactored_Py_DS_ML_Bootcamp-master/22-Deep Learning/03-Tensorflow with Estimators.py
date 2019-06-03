#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Tensorflow with Estimators
# 
# As we saw previously how to build a full Multi-Layer Perceptron model with full Sessions in Tensorflow. Unfortunately this was an extremely involved process. However developers have created Estimators that have an easier to use flow!
# 
# It is much easier to use, but you sacrifice some level of customization of your model. Let's go ahead and explore it!

# ## Get the Data
# 
# We will the iris data set.
# 
# Let's get the data:

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


df.head()


# In[4]:


df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']


# In[5]:


X = df.drop('target',axis=1)
y = df['target'].apply(int)


# ## Train Test Split

# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Estimators
# 
# Let's show you how to use the simpler Estimator interface!

# In[8]:


import tensorflow as tf


# ## Feature Columns

# In[9]:


X.columns


# In[10]:


feat_cols = []

for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))


# In[11]:


feat_cols


# ## Input Function

# In[12]:


# there is also a pandas_input_fn we'll see in the exercise!!
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)


# In[13]:


classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3,feature_columns=feat_cols)


# In[14]:


classifier.train(input_fn=input_func,steps=50)


# ## Model Evaluation
# 
# ** Use the predict method from the classifier model to create predictions from X_test **

# In[15]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[16]:


note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[17]:


note_predictions[0]


# In[18]:


final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# ** Now create a classification report and a Confusion Matrix. Does anything stand out to you?**

# In[19]:


from sklearn.metrics import classification_report,confusion_matrix


# In[20]:


print(confusion_matrix(y_test,final_preds))


# In[21]:


print(classification_report(y_test,final_preds))


# # Great Job!
