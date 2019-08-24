#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Seaborn Exercises
# 
# Time to practice your new seaborn skills! Try to recreate the plots below (don't worry about color schemes, just the plot itself.

# ## The Data
# 
# We will be working with a famous titanic data set for these exercises. Later on in the Machine Learning section of the course, we will revisit this data, and use it to predict survival rates of passengers. For now, we'll just focus on the visualization of the data with seaborn:

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.set_style('whitegrid')


# In[3]:


titanic = sns.load_dataset('titanic')


# In[4]:


titanic.head()


# # Exercises
# 
# **Recreate the plots below using the titanic dataframe. There are very few hints since most of the plots can be done with just one or two lines of code and a hint would basically give away the solution. Keep careful attention to the x and y labels for hints.**
# 
# **Note! In order to not lose the plot image, make sure you don't code in the cell that is directly above the plot, there is an extra cell above that one which won't overwrite that plot!**

# In[42]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[46]:


get_ipython().run_line_magic('pinfo', 'sns.jointplot')


# In[7]:


sns.jointplot(
    x='fare', 
    y='age', 
    data=titanic, 
    kind='scatter'
)


# In[41]:





# In[43]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[47]:


get_ipython().run_line_magic('pinfo', 'sns.distplot')


# In[13]:


sns.distplot(
    a=titanic['fare'],
    bins=30,
    kde=None
)


# In[44]:





# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[48]:


get_ipython().run_line_magic('pinfo', 'sns.boxplot')


# In[17]:


sns.boxplot(x='class', y='age', data=titanic)


# In[45]:





# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[49]:


get_ipython().run_line_magic('pinfo', 'sns.swarmplot')


# In[21]:


sns.swarmplot(x='class', y='age', data=titanic)


# In[46]:





# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[50]:


get_ipython().run_line_magic('pinfo', 'sns.countplot')


# In[28]:


sns.countplot(
    x='sex', 
    data=titanic
)


# In[47]:





# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[51]:


get_ipython().run_line_magic('pinfo', 'sns.heatmap')


# In[34]:


tita_cor = titanic.corr()
sns.heatmap(tita_cor)


# In[48]:





# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[52]:


get_ipython().run_line_magic('pinfo', 'sns.FacetGrid')


# In[45]:


ft = sns.FacetGrid(
    data=titanic,
    col='sex'
)
ft.map(sns.distplot, 'age', kde=None)


# In[49]:





# # Great Job!
# 
# ### That is it for now! We'll see a lot more of seaborn practice problems in the machine learning section!
