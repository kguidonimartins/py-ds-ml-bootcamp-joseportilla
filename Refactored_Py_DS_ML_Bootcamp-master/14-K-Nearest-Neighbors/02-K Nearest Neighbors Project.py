#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # K Nearest Neighbors Project 
# 
# Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.
# ## Import Libraries
# **Import pandas,seaborn, and the usual libraries.**

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ## Get the Data
# 
# **Read the 'KNN_Project_Data csv file into a dataframe**

# In[43]:


df = pd.read_csv('KNN_Project_Data')
df.columns = map(str.lower, df.columns)
df.columns = df.columns.str.replace(" ", "_")


# **Check the head of the dataframe.**

# In[44]:


df.head()


# In[ ]:





# # EDA
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[45]:


sns.pairplot(df, hue='target_class')


# In[ ]:





# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# **Import StandardScaler from Scikit learn.**

# In[ ]:





# **Create a StandardScaler() object called scaler.**

# In[46]:


scaler = StandardScaler()


# In[ ]:





# **Fit scaler to the features.**

# In[47]:


scaler.fit(df.drop('target_class', axis=1))


# In[ ]:





# **Use the .transform() method to transform the features to a scaled version.**

# In[48]:


scaled_features = scaler.transform(df.drop('target_class', axis=1))
scaled_features


# In[ ]:





# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[49]:


df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()


# In[50]:


# check mean
df_feat.mean().round()


# In[51]:


# check sd
df_feat.std().round()


# In[ ]:





# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[52]:


X = df_feat
y = df['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[ ]:





# In[ ]:





# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**

# In[ ]:





# **Create a KNN model instance with n_neighbors=1**

# In[53]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:





# **Fit this KNN model to the training data.**

# In[54]:


knn.fit(X_train, y_train)


# In[ ]:





# # Predictions and Evaluations
# Let's evaluate our KNN model!

# **Use the predict method to predict values using your KNN model and X_test.**

# In[55]:


pred = knn.predict(X_test)


# In[ ]:





# **Create a confusion matrix and classification report.**

# In[56]:


print(confusion_matrix(y_test, pred))


# In[ ]:





# In[ ]:





# In[57]:


print(classification_report(y_test, pred))


# In[ ]:





# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# **Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# In[58]:


error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))    


# In[ ]:





# **Now create the following plot using the information from your for loop.**

# In[59]:


plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error rate')


# In[ ]:





# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

# In[61]:


knn = KNeighborsClassifier(n_neighbors=26)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[ ]:





# # Great Job!
