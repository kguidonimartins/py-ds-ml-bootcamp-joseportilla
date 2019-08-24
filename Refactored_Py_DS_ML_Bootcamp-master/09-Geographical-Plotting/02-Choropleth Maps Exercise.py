#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Choropleth Maps Exercise 
# 
# Welcome to the Choropleth Maps Exercise! In this exercise we will give you some simple datasets and ask you to create Choropleth Maps from them. Due to the Nature of Plotly we can't show you examples
# 
# [Full Documentation Reference](https://plot.ly/python/reference/#choropleth)
# 
# ## Plotly Imports

# In[1]:


import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import pandas as pd


# **Import pandas and read the csv file: 2014_World_Power_Consumption**

# In[2]:


df = pd.read_csv('2014_World_GDP')


# **Check the head of the DataFrame.**

# In[3]:


df.head()


# **Referencing the lecture notes, create a Choropleth Plot of the Power Consumption for Countries using the data and layout dictionary.**

# In[4]:


data = dict(
    type = "choropleth",
    locations = df['CODE'],
    text = df['COUNTRY'],
    z = df['GDP (BILLIONS)'],
    colorbar = {'title': 'GDP in Billions USD'}
    )

layout = dict(
    title = '2014 Global GDP',
    geo = dict(
        showframe = False,
        # see at: https://plot.ly/python/reference/#layout-geo-projection
        projection = {'type': 'stereographic'}
        )
    )


# In[5]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap, validate=False)


# ## USA Choropleth
# 
# **Import the 2012_Election_Data csv file using pandas.**

# In[6]:


df = pd.read_csv('2012_Election_Data')


# **Check the head of the DataFrame.**

# In[7]:


df.head()


# **Now create a plot that displays the Voting-Age Population (VAP) per state. If you later want to play around with other columns, make sure you consider their data type. VAP has already been transformed to a float for you.**

# In[20]:


data = dict(
    type='choropleth',
    colorscale = 'Viridis',
    reversescale = True,
    locations = df['State Abv'],
    z = df['Voting-Age Population (VAP)'],
    locationmode = 'USA-states',
    text = df['State'],
    marker = dict(
        line = dict(
            color = 'rgb(255,255,255)',
            width = 1
        )
    ),
    colorbar = {
        'title':"Voting-Age Population (VAP)"
    }
) 

layout = dict(
    title = '2012 General Election Voting Data',
    geo = dict(
        scope='usa',
        showlakes = True,
        lakecolor = 'rgb(85,173,240)'
    )
)


# In[21]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# # Great Job!
