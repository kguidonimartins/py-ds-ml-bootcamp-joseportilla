#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Random Forest Project 
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Import Libraries
# 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# ## Get the Data
# 
# **Use pandas to read loan_data.csv as a dataframe called loans.**

# In[2]:


df = pd.read_csv('loan_data.csv')


# **Check out the info(), head(), and describe() methods on loans.**

# In[3]:


df.info()


# In[4]:





# In[5]:


df.describe()


# In[5]:





# In[6]:


df.head()


# In[6]:





# # Exploratory Data Analysis
# 
# Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.
# 
# **Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# *Note: This is pretty tricky, feel free to reference the solutions. You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()*

# In[14]:


plt.figure(figsize=(10,6))
df[df['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
df[df['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[7]:





# **Create a similar figure, except this time select by the not.fully.paid column.**

# In[15]:


plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[8]:





# **Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.**

# In[20]:


plt.figure(figsize=(12,8))
sns.countplot(x='purpose', data=df, hue='not.fully.paid')


# In[9]:





# **Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**

# In[26]:


sns.jointplot(x='fico', y='int.rate', data=df, alpha=.1, color='black')


# In[10]:





# **Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**

# In[28]:


sns.lmplot(x='fico', y='int.rate', data=df, hue='not.fully.paid')


# In[31]:


sns.lmplot(x='fico', y='int.rate', data=df, col='not.fully.paid')


# In[11]:





# # Setting up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**

# In[32]:


df.info()


# In[12]:





# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# 
# **Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**

# In[35]:


cat_feats = ['purpose']


# In[13]:





# **Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.**

# In[37]:


df_dummy = pd.get_dummies(df, columns=cat_feats, drop_first=True)


# In[14]:





# In[ ]:





# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# **Use sklearn to split your data into a training set and a testing set as we've done in the past.**

# In[52]:


response = 'not.fully.paid'
X = df_dummy.drop(response, axis=1)
y = df_dummy[response]


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[16]:





# In[17]:





# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# **Import DecisionTreeClassifier**

# In[18]:


from sklearn.tree import DecisionTreeClassifier


# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**

# In[54]:


dtree = DecisionTreeClassifier()


# In[19]:





# In[55]:


dtree.fit(X_train, y_train)


# In[32]:





# ## Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**

# In[56]:


pred_dtree = dtree.predict(X_test)


# In[57]:


print(confusion_matrix(y_test, pred_dtree))


# In[58]:


print(classification_report(y_test, pred_dtree))


# In[21]:





# In[22]:





# In[23]:





# In[24]:





# ## Training the Random Forest model
# 
# Now its time to train our model!
# 
# **Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**

# In[49]:


rfc = RandomForestClassifier()


# In[50]:


rfc.fit(X_train, y_train)


# In[25]:





# In[26]:





# In[27]:





# ## Predictions and Evaluation
# 
# Let's predict off the y_test values and evaluate our model.
# 
# **Predict the class of not.fully.paid for the X_test data.**

# In[60]:


rfc_pred = rfc.predict(X_test)


# In[28]:





# **Now create a classification report from the results. Do you get anything strange or some sort of warning?**

# In[61]:


print(confusion_matrix(y_test, rfc_pred))


# In[62]:


print(classification_report(y_test, rfc_pred))


# In[29]:





# In[30]:





# **Show the Confusion Matrix for the predictions.**

# In[31]:





# **What performed better the random forest or the decision tree?**

# That depends. But the recall rate for both models is too low.

# # Great Job!
