#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[8]:


df = pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[9]:


df.head()


# **Use info and describe() on ad_data**

# In[10]:


df.info()


# In[41]:





# In[11]:


df.describe()


# In[42]:





# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# **Create a histogram of the Age**

# In[15]:


sns.distplot(a=df['Age'], bins=30, kde=False)


# In[48]:





# **Create a jointplot showing Area Income versus Age.**

# In[17]:


sns.jointplot(x='Age', y='Area Income', data=df)


# In[64]:





# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[22]:


sns.jointplot(x='Age', y='Daily Time Spent on Site', data=df, kind='kde')


# In[66]:





# **Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[24]:


sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=df)


# In[72]:





# **Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[25]:


sns.pairplot(data=df, hue='Clicked on Ad')


# In[84]:





# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# **Split the data into training set and testing set using train_test_split**

# In[27]:


df.columns


# In[35]:


city = pd.get_dummies(df['City'], drop_first=True)
country = pd.get_dummies(df['Country'], drop_first=True)
df = pd.concat([df, city, country], axis=1)


# In[37]:


df.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)


# In[38]:


df.head()


# In[39]:


X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# **Train and fit a logistic regression model on the training set.**

# In[41]:


logmodel = LogisticRegression()


# In[42]:


logmodel.fit(X_train, y_train)


# In[91]:





# In[92]:





# ## Predictions and Evaluations
# **Now predict values for the testing data.**

# In[43]:


predictions = logmodel.predict(X_test)


# In[94]:





# **Create a classification report for the model.**

# In[44]:


print(classification_report(y_test, predictions))


# In[95]:





# In[96]:





# ## Great Job!
