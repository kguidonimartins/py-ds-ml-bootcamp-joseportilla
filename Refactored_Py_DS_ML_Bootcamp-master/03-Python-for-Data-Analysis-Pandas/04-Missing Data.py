#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Missing Data
# 
# Let's show a few convenient methods to deal with Missing Data in pandas:

# In[1]:


import numpy as np
import pandas as pd


# In[9]:


df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})


# In[10]:


df


# In[12]:


df.dropna()


# In[13]:


df.dropna(axis=1)


# In[14]:


df.dropna(thresh=2)


# In[15]:


df.fillna(value='FILL VALUE')


# In[17]:


df['A'].fillna(value=df['A'].mean())


# # Great Job!
