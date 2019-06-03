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

# In[1]:





# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[2]:





# **Check the head of the dataframe.**

# In[23]:





# # EDA
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[4]:





# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[5]:





# ** Create a StandardScaler() object called scaler.**

# In[6]:





# ** Fit scaler to the features.**

# In[7]:





# **Use the .transform() method to transform the features to a scaled version.**

# In[8]:





# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[9]:





# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[10]:





# In[11]:





# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**

# In[12]:





# **Create a KNN model instance with n_neighbors=1**

# In[13]:





# **Fit this KNN model to the training data.**

# In[14]:





# # Predictions and Evaluations
# Let's evaluate our KNN model!

# **Use the predict method to predict values using your KNN model and X_test.**

# In[24]:





# ** Create a confusion matrix and classification report.**

# In[16]:





# In[17]:





# In[18]:





# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# In[25]:





# **Now create the following plot using the information from your for loop.**

# In[20]:





# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

# In[21]:





# # Great Job!
