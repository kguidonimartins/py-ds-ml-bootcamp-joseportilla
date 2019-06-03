#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Grids
# 
# Grids are general types of plots that allow you to map plot types to rows and columns of a grid, this helps you create similar plots separated by features.

# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


iris = sns.load_dataset('iris')


# In[28]:


iris.head()


# ## PairGrid
# 
# Pairgrid is a subplot grid for plotting pairwise relationships in a dataset.

# In[25]:


# Just the Grid
sns.PairGrid(iris)


# In[26]:


# Then you map to the grid
g = sns.PairGrid(iris)
g.map(plt.scatter)


# In[30]:


# Map to upper,lower, and diagonal
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


# ## pairplot
# 
# pairplot is a simpler version of PairGrid (you'll use quite often)

# In[31]:


sns.pairplot(iris)


# In[33]:


sns.pairplot(iris,hue='species',palette='rainbow')


# ## Facet Grid
# 
# FacetGrid is the general way to create grids of plots based off of a feature:

# In[34]:


tips = sns.load_dataset('tips')


# In[35]:


tips.head()


# In[36]:


# Just the Grid
g = sns.FacetGrid(tips, col="time", row="smoker")


# In[37]:


g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")


# In[42]:


g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')
# Notice hwo the arguments come after plt.scatter call
g = g.map(plt.scatter, "total_bill", "tip").add_legend()


# ## JointGrid
# 
# JointGrid is the general version for jointplot() type grids, for a quick example:

# In[43]:


g = sns.JointGrid(x="total_bill", y="tip", data=tips)


# In[45]:


g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(sns.regplot, sns.distplot)


# Reference the documentation as necessary for grid types, but most of the time you'll just use the easier plots discussed earlier.
# # Great Job!
