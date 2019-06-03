#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # NumPy Operations

# ## Arithmetic
# 
# You can easily perform array with array arithmetic, or scalar with array arithmetic. Let's see some examples:

# In[1]:


import numpy as np
arr = np.arange(0,10)


# In[2]:


arr + arr


# In[3]:


arr * arr


# In[4]:


arr - arr


# In[5]:


# Warning on division by zero, but not an error!
# Just replaced with nan
arr/arr


# In[6]:


# Also warning, but not an error instead infinity
1/arr


# In[10]:


arr**3


# ## Universal Array Functions
# 
# Numpy comes with many [universal array functions](http://docs.scipy.org/doc/numpy/reference/ufuncs.html), which are essentially just mathematical operations you can use to perform the operation across the array. Let's show some common ones:

# In[12]:


#Taking Square Roots
np.sqrt(arr)


# In[13]:


#Calcualting exponential (e^)
np.exp(arr)


# In[14]:


np.max(arr) #same as arr.max()


# In[15]:


np.sin(arr)


# In[16]:


np.log(arr)


# # Great Job!
# 
# That's all we need to know for now!
