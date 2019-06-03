#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # NumPy Indexing and Selection
# 
# In this lecture we will discuss how to select elements or groups of elements from an array.

# In[2]:


import numpy as np


# In[3]:


#Creating sample array
arr = np.arange(0,11)


# In[4]:


#Show
arr


# ## Bracket Indexing and Selection
# The simplest way to pick one or some elements of an array looks very similar to python lists:

# In[5]:


#Get a value at an index
arr[8]


# In[6]:


#Get values in a range
arr[1:5]


# In[7]:


#Get values in a range
arr[0:5]


# ## Broadcasting
# 
# Numpy arrays differ from a normal Python list because of their ability to broadcast:

# In[8]:


#Setting a value with index range (Broadcasting)
arr[0:5]=100

#Show
arr


# In[9]:


# Reset array, we'll see why I had to reset in  a moment
arr = np.arange(0,11)

#Show
arr


# In[10]:


#Important notes on Slices
slice_of_arr = arr[0:6]

#Show slice
slice_of_arr


# In[11]:


#Change Slice
slice_of_arr[:]=99

#Show Slice again
slice_of_arr


# Now note the changes also occur in our original array!

# In[12]:


arr


# Data is not copied, it's a view of the original array! This avoids memory problems!

# In[13]:


#To get a copy, need to be explicit
arr_copy = arr.copy()

arr_copy


# ## Indexing a 2D array (matrices)
# 
# The general format is **arr_2d[row][col]** or **arr_2d[row,col]**. I recommend usually using the comma notation for clarity.

# In[14]:


arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

#Show
arr_2d


# In[15]:


#Indexing row
arr_2d[1]


# In[16]:


# Format is arr_2d[row][col] or arr_2d[row,col]

# Getting individual element value
arr_2d[1][0]


# In[17]:


# Getting individual element value
arr_2d[1,0]


# In[18]:


# 2D array slicing

#Shape (2,2) from top right corner
arr_2d[:2,1:]


# In[19]:


#Shape bottom row
arr_2d[2]


# In[20]:


#Shape bottom row
arr_2d[2,:]


# ### Fancy Indexing
# 
# Fancy indexing allows you to select entire rows or columns out of order,to show this, let's quickly build out a numpy array:

# In[21]:


#Set up matrix
arr2d = np.zeros((10,10))


# In[22]:


#Length of array
arr_length = arr2d.shape[1]


# In[23]:


#Set up array

for i in range(arr_length):
    arr2d[i] = i
    
arr2d


# Fancy indexing allows the following

# In[24]:


arr2d[[2,4,6,8]]


# In[25]:


#Allows in any order
arr2d[[6,4,2,7]]


# ## More Indexing Help
# Indexing a 2d matrix can be a bit confusing at first, especially when you start to add in step size. Try google image searching NumPy indexing to fins useful images, like this one:
# 
# <img src= 'http://memory.osu.edu/classes/python/_images/numpy_indexing.png' width=500/>

# ## Selection
# 
# Let's briefly go over how to use brackets for selection based off of comparison operators.

# In[28]:


arr = np.arange(1,11)
arr


# In[30]:


arr > 4


# In[31]:


bool_arr = arr>4


# In[32]:


bool_arr


# In[33]:


arr[bool_arr]


# In[34]:


arr[arr>2]


# In[37]:


x = 2
arr[arr>x]


# # Great Job!
# 
