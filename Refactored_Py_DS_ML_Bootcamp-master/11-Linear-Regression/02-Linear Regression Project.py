#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Linear Regression Project
# 
# Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out! Let's get started!
# 
# Just follow the steps below to analyze the customer data (it's fake, don't worry I didn't give you real credit card numbers or emails).

# ## Imports
# **Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (You'll import sklearn as you need it.)**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import metrics


# In[275]:





# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# **Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[3]:


df = pd.read_csv('Ecommerce Customers')


# In[276]:





# **Check the head of customers, and check out its info() and describe() methods.**

# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[277]:





# In[278]:





# In[279]:





# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# For the rest of the exercise we'll only be using the numerical data of the csv file.
# ___
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[9]:


sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)


# In[280]:





# In[281]:





# **Do the same but with the Time on App column instead.**

# In[10]:


sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)


# In[282]:





# **Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[11]:


sns.jointplot(x='Time on Website', y='Length of Membership', data=df, kind='hex')


# In[283]:





# **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.(Don't worry about the the colors)**

# In[12]:


sns.pairplot(df)


# In[284]:





# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# Length of Membership!

# In[285]:





# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership.**

# In[13]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)


# In[286]:





# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.  
# **Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column.**

# In[17]:


X = df.select_dtypes(include=['float64', 'int'])
X = X.drop('Yearly Amount Spent', axis=1)
X.head()


# In[19]:


y = df['Yearly Amount Spent']
y.head()


# In[287]:





# In[288]:





# **Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)


# In[42]:


X_train.shape


# In[43]:


X_test.shape


# In[44]:


y_train.shape


# In[45]:


y_test.shape


# In[289]:





# In[290]:





# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# **Import LinearRegression from sklearn.linear_model**

# In[291]:





# **Create an instance of a LinearRegression() model named lm.**

# In[27]:


lm = LinearRegression()


# In[292]:





# **Train/fit lm on the training data.**

# In[46]:


lm.fit(X_train, y_train)


# In[293]:





# **Print out the coefficients of the model**

# In[47]:


print('Coefficients:', lm.coef_)


# In[294]:





# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# **Use lm.predict() to predict off the X_test set of the data.**

# In[48]:


predictions = lm.predict(X_test)


# In[295]:





# **Create a scatterplot of the real test values versus the predicted values.**

# In[49]:


plt.scatter(y_test, predictions)


# In[296]:





# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# **Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[77]:


print('R^2:', round(metrics.r2_score(y_test, predictions), 3))


# In[50]:


print('MAE:', round(metrics.mean_absolute_error(y_test, predictions), 2))
print('MSE:', round(metrics.mean_squared_error(y_test, predictions), 2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, predictions)), 2))


# In[303]:





# ## Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[51]:


sns.distplot((y_test - predictions))


# In[317]:





# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# **Recreate the dataframe below.**

# In[69]:


df_coef = pd.DataFrame(
    data=np.round(lm.coef_, 2), 
    index=X.columns, 
    columns=['Coefficient']
)
df_coef.sort_values(
    by='Coefficient', 
    ascending=False
)


# In[298]:





# **How can you interpret these coefficients?**

# 'Length of Membership' has the highest effect on the 'Yearly Amount Spent', followed by the 'Time on App', and the 'Avg. Session Length'. 'Time on website' has the lowest effect on the 'Yearly Amount Spent'. 
# 
# <!-- 
# TODO: Test interactions among the features. 
# -->

# **Do you think the company should focus more on their mobile app or on their website?**

# Our model show that the company should focus more on their mobile app.

# ## Great Job!
# 
# Congrats on your contract work! The company loved the insights! Let's move on.
