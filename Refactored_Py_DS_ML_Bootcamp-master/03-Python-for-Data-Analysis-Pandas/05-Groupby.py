#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Groupby
# 
# The groupby method allows you to group rows of data together and call aggregate functions

# In[31]:


import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}


# In[32]:


df = pd.DataFrame(data)


# In[33]:


df


# ** Now you can use the .groupby() method to group rows together based off of a column name. For instance let's group based off of Company. This will create a DataFrameGroupBy object:**

# In[34]:


df.groupby('Company')


# You can save this object as a new variable:

# In[35]:


by_comp = df.groupby("Company")


# And then call aggregate methods off the object:

# In[36]:


by_comp.mean()


# In[37]:


df.groupby('Company').mean()


# More examples of aggregate methods:

# In[38]:


by_comp.std()


# In[39]:


by_comp.min()


# In[40]:


by_comp.max()


# In[41]:


by_comp.count()


# In[42]:


by_comp.describe()


# In[43]:


by_comp.describe().transpose()


# In[44]:


by_comp.describe().transpose()['GOOG']


# # Great Job!
