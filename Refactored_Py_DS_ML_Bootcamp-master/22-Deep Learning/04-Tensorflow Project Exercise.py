#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Tensorflow Project Exercise
# Let's wrap up this Deep Learning by taking a a quick look at the effectiveness of Neural Nets!
# 
# We'll use the [Bank Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) from the UCI repository.
# 
# The data consists of 5 columns:
# 
# * variance of Wavelet Transformed image (continuous)
# * skewness of Wavelet Transformed image (continuous)
# * curtosis of Wavelet Transformed image (continuous)
# * entropy of image (continuous)
# * class (integer)
# 
# Where class indicates whether or not a Bank Note was authentic.
# 
# This sort of task is perfectly suited for Neural Networks and Deep Learning! Just follow the instructions below to get started!

# ## Get the Data
# 
# ** Use pandas to read in the bank_note_data.csv file **

# In[1]:





# In[2]:





# ** Check the head of the Data **

# In[3]:





# ## EDA
# 
# We'll just do a few quick plots of the data.
# 
# ** Import seaborn and set matplolib inline for viewing **

# In[4]:





# ** Create a Countplot of the Classes (Authentic 1 vs Fake 0) **

# In[5]:





# ** Create a PairPlot of the Data with Seaborn, set Hue to Class **

# In[6]:





# ## Data Preparation 
# 
# When using Neural Network and Deep Learning based systems, it is usually a good idea to Standardize your data, this step isn't actually necessary for our particular data set, but let's run through it for practice!
# 
# ### Standard Scaling
# 
# 

# In[7]:





# **Create a StandardScaler() object called scaler.**

# In[8]:





# **Fit scaler to the features.**

# In[9]:





# **Use the .transform() method to transform the features to a scaled version.**

# In[10]:





# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[11]:





# ## Train Test Split
# 
# ** Create two objects X and y which are the scaled feature values and labels respectively.**

# In[12]:





# In[13]:





# ** Use SciKit Learn to create training and testing sets of the data as we've done in previous lectures:**

# In[14]:





# In[15]:





# # Tensorflow

# In[16]:





# ** Create a list of feature column objects using tf.feature.numeric_column() as we did in the lecture**

# In[17]:





# In[18]:





# In[19]:





# ** Create an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10] hidden unit layer structure:**

# In[20]:





# ** Now create a tf.estimator.pandas_input_fn that takes in your X_train, y_train, batch_size and set shuffle=True. You can play around with the batch_size parameter if you want, but let's start by setting it to 20 since our data isn't very big. **

# In[21]:





# ** Now train classifier to the input function. Use steps=500. You can play around with these values if you want!**
# 
# *Note: Ignore any warnings you get, they won't effect your output*

# In[22]:





# ## Model Evaluation

# ** Create another pandas_input_fn that takes in the X_test data for x. Remember this one won't need any y_test info since we will be using this for the network to create its own predictions. Set shuffle=False since we don't need to shuffle for predictions.**

# In[23]:





# ** Use the predict method from the classifier model to create predictions from X_test **

# In[24]:





# In[25]:





# In[26]:





# ** Now create a classification report and a Confusion Matrix. Does anything stand out to you?**

# In[27]:





# In[28]:





# In[29]:





# ## Optional Comparison
# 
# ** You should have noticed extremely accurate results from the DNN model. Let's compare this to a Random Forest Classifier for a reality check!**
# 
# **Use SciKit Learn to Create a Random Forest Classifier and compare the confusion matrix and classification report to the DNN model**

# In[30]:





# In[31]:





# In[32]:





# In[33]:





# In[34]:





# In[35]:





# ** It should have also done very well, possibly perfect! Hopefully you have seen the power of DNN! **

# # Great Job!
