#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!

# ## Data and Setup

# **Import numpy and pandas**

# In[1]:


import pandas as pd
import numpy as np


# In[129]:





# **Import visualization libraries and set %matplotlib inline.**

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[130]:





# **Read in the csv file as a dataframe called df**

# In[3]:


df = pd.read_csv("911.csv")


# In[131]:





# **Check the info() of the df**

# In[4]:


df.info()


# In[132]:





# **Check the head of df**

# In[4]:


df.head()


# In[155]:





# ## Basic Questions

# **What are the top 5 zipcodes for 911 calls?**

# In[22]:


df['zip'].value_counts().head()


# In[134]:





# **What are the top 5 townships (twp) for 911 calls?**

# In[24]:


df['twp'].value_counts().head()


# In[135]:





# **Take a look at the 'title' column, how many unique title codes are there?**

# In[36]:


df['title'].nunique()


# In[136]:





# ## Creating new features

# **In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.**

# In[5]:


df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])


# In[6]:


df.head()


# In[137]:





# **What is the most common Reason for a 911 call based off of this new column?**

# In[7]:


df['Reason'].value_counts().index


# In[138]:





# **Now use seaborn to create a countplot of 911 calls by Reason.**

# In[8]:


sns.countplot('Reason', data=df, order=df['Reason'].value_counts().index)


# In[139]:





# **Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?**

# In[9]:


df['timeStamp'].dtype


# In[10]:


type(df['timeStamp'].iloc[0])


# In[140]:





# **You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects.**

# In[12]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[184]:





# **You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[13]:


time = df['timeStamp']
df['Hour'] = time.dt.hour
df['Month'] = time.dt.month
df['Day of Week'] = time.dt.dayofweek


# In[142]:





# **Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:**
# 
# `dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}`

# In[14]:


dmap = {
    0: 'Mon',
    1: 'Tue',
    2: 'Wed',
    3: 'Thu',
    4: 'Fri', 
    5: 'Sat', 
    6: 'Sun'
}
df['Day of Week'] = df['Day of Week'].map(dmap)
df.head()


# In[144]:





# **Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.**

# In[15]:


sns.countplot(x='Day of Week', data=df, hue='Reason').legend(loc=0)


# In[168]:





# **Now do the same for Month:**

# In[16]:


sns.countplot(x='Month', data=df, hue='Reason').legend(loc=0)


# In[3]:





# **Did you notice something strange about the Plot?**
# 
# _____
# 
# **You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas...**

# **Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.**

# In[17]:


by_month = df.groupby('Month').count()
by_month.head()


# In[169]:





# **Now create a simple plot off of the dataframe indicating the count of calls per month.**

# In[18]:


by_month.plot(y='twp', kind='line')


# In[175]:





# **Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column.**

# In[19]:


sns.lmplot(x='Month', y='twp', data=by_month.reset_index())


# In[187]:





# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.** 

# In[75]:


df['Date'] = df['timeStamp'].apply(lambda x: x.date())


# In[193]:





# **Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[79]:


by_date = df.groupby('Date').count()
by_date['twp'].plot(figsize = (12, 4))


# In[197]:





# **Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[93]:


by_reason = df.groupby(['Date', 'Reason']).count().reset_index()


# In[97]:


by_reason[by_reason['Reason'] == 'Traffic'].plot(x='Date', y='twp')


# In[199]:





# In[98]:


by_reason[by_reason['Reason'] == 'Fire'].plot(x='Date', y='twp')


# In[201]:





# In[99]:


by_reason[by_reason['Reason'] == 'EMS'].plot(x='Date', y='twp')


# In[202]:





# ____
# **Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[42]:


hours_by_dow = df.groupby(['Day of Week', 'Hour']).count()['Reason'].unstack()
hours_by_dow


# In[203]:





# **Now create a HeatMap using this new DataFrame.**

# In[28]:


plt.figure(figsize=(12, 6))
sns.heatmap(hours_by_dow, cmap='viridis')


# In[204]:





# **Now create a clustermap using this DataFrame.**

# In[29]:


sns.clustermap(hours_by_dow, cmap='viridis')


# In[205]:





# **Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.**

# In[38]:


month_by_dow = df.groupby(['Day of Week', 'Month']).count()['Reason'].unstack()
month_by_dow


# In[207]:





# In[40]:


plt.figure(figsize=(12, 6))
sns.heatmap(month_by_dow, cmap='viridis')


# In[208]:





# In[41]:


sns.clustermap(month_by_dow, cmap='viridis')


# In[209]:





# **Continue exploring the Data however you see fit!**
# # Great Job!
