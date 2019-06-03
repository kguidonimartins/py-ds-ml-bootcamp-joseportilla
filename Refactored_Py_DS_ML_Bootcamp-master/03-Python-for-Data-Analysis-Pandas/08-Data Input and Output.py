#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Data Input and Output
# 
# This notebook is the reference code for getting input and output, pandas can read a variety of file types using its pd.read_ methods. Let's take a look at the most common data types:

# In[1]:


import numpy as np
import pandas as pd


# ## CSV
# 
# ### CSV Input

# In[25]:


df = pd.read_csv('example')
df


# ### CSV Output

# In[24]:


df.to_csv('example',index=False)


# ## Excel
# Pandas can read and write excel files, keep in mind, this only imports data. Not formulas or images, having images or macros may cause this read_excel method to crash. 

# ### Excel Input

# In[35]:


pd.read_excel('Excel_Sample.xlsx',sheetname='Sheet1')


# ### Excel Output

# In[33]:


df.to_excel('Excel_Sample.xlsx',sheet_name='Sheet1')


# ## HTML
# 
# You may need to install htmllib5,lxml, and BeautifulSoup4. In your terminal/command prompt run:
# 
#     conda install lxml
#     conda install html5lib
#     conda install BeautifulSoup4
# 
# Then restart Jupyter Notebook.
# (or use pip install if you aren't using the Anaconda Distribution)
# 
# Pandas can read table tabs off of html. For example:

# ### HTML Input
# 
# Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects:

# In[5]:


df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')


# In[7]:


df[0]


# ____

# _____
# _____
# # SQL (Optional)
# 
# * Note: If you are completely unfamiliar with SQL you can check out my other course: "Complete SQL Bootcamp" to learn SQL.

# The pandas.io.sql module provides a collection of query wrappers to both facilitate data retrieval and to reduce dependency on DB-specific API. Database abstraction is provided by SQLAlchemy if installed. In addition you will need a driver library for your database. Examples of such drivers are psycopg2 for PostgreSQL or pymysql for MySQL. For SQLite this is included in Python’s standard library by default. You can find an overview of supported drivers for each SQL dialect in the SQLAlchemy docs.
# 
# 
# If SQLAlchemy is not installed, a fallback is only provided for sqlite (and for mysql for backwards compatibility, but this is deprecated and will be removed in a future version). This mode requires a Python database adapter which respect the Python DB-API.
# 
# See also some cookbook examples for some advanced strategies.
# 
# The key functions are:
# 
# * read_sql_table(table_name, con[, schema, ...])	
#     * Read SQL database table into a DataFrame.
# * read_sql_query(sql, con[, index_col, ...])	
#     * Read SQL query into a DataFrame.
# * read_sql(sql, con[, index_col, ...])	
#     * Read SQL query or database table into a DataFrame.
# * DataFrame.to_sql(name, con[, flavor, ...])	
#     * Write records stored in a DataFrame to a SQL database.

# In[36]:


from sqlalchemy import create_engine


# In[37]:


engine = create_engine('sqlite:///:memory:')


# In[40]:


df.to_sql('data', engine)


# In[42]:


sql_df = pd.read_sql('data',con=engine)


# In[43]:


sql_df


# # Great Job!
