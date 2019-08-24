#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Pandas Data Visualization Exercise
# 
# This is just a quick exercise for you to review the various plots we showed earlier. Use **df3** to replicate the following plots. 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('df3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df3.info()


# In[3]:


df3.head()


# **Recreate this scatter plot of b vs a. Note the color and size of the points. Also note the figure size. See if you can figure out how to stretch it in a similar fashion. Remeber back to your matplotlib lecture...**

# In[36]:


df3.plot.scatter(
    x='a',
    y='b',
    c='red',
    figsize=[12, 3],
    xlim=[-0.2, 1.2],
    ylim=[-0.2, 1.2],
    s=50,
    marker='o'
)


# In[4]:





# **Create a histogram of the 'a' column.**

# In[44]:


df3.hist(column='a')


# In[5]:





# **These plots are okay, but they don't look very polished. Use style sheets to set the style to 'ggplot' and redo the histogram from above. Also figure out how to add more bins to it.**

# In[47]:


plt.style.use('ggplot')


# In[48]:


df3.hist(column='a', alpha=0.3, bins=30)


# In[7]:





# **Create a boxplot comparing the a and b columns.**

# In[49]:


df3[['a', 'b']].boxplot()


# In[8]:





# **Create a kde plot of the 'd' column.**

# In[53]:


df3['d'].plot(kind='kde')


# In[9]:





# **Figure out how to increase the linewidth and make the linestyle dashed. (Note: You would usually not dash a kde plot line)**

# In[63]:


df3['d'].plot.density(
    linewidth=5, 
    linestyle='dashed'
)


# In[10]:





# **Create an area plot of all the columns for just the rows up to 30. (hint: use .ix).**

# In[69]:


df3.loc[0:30].plot.area(alpha=0.4)


# In[15]:





# ## Bonus Challenge!
# Note, you may find this really hard, reference the solutions if you can't figure it out!
# 
# **Notice how the legend in our previous figure overlapped some of actual diagram. Can you figure out how to display the legend outside of the plot as shown below?**
# 
# **Try searching Google for a good stackoverflow link on this topic. If you can't find it on your own - [use this one for a hint.](http://stackoverflow.com/questions/23556153/how-to-put-legend-outside-the-plot-with-pandas)**

# In[71]:


df3.loc[0:30].plot.area(alpha=0.4).legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ref: https://stackoverflow.com/a/4701285/5974372


# In[17]:





# # Great Job!
