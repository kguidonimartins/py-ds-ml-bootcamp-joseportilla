#!/usr/bin/env python
# coding: utf-8

# # Introduction to Spark and Python
# 
# Let's learn how to use Spark with Python by using the pyspark library! Make sure to view the video lecture explaining Spark and RDDs before continuing on with this code.
# 
# This notebook will serve as reference code for the Big Data section of the course involving Amazon Web Services. The video will provide fuller explanations for what the code is doing.
# 
# ## Creating a SparkContext
# 
# First we need to create a SparkContext. We will import this from pyspark:

# In[1]:


from pyspark import SparkContext


# Now create the SparkContext,A SparkContext represents the connection to a Spark cluster, and can be used to create an RDD and broadcast variables on that cluster.
# 
# *Note! You can only have one SparkContext at a time the way we are running things here.*

# In[2]:


sc = SparkContext()


# ## Basic Operations
# 
# We're going to start with a 'hello world' example, which is just reading a text file. First let's create a text file.
# ___

# Let's write an example text file to read, we'll use some special jupyter notebook commands for this, but feel free to use any .txt file:

# In[9]:


get_ipython().run_cell_magic('writefile', 'example.txt', 'first line\nsecond line\nthird line\nfourth line')


# ### Creating the RDD

# Now we can take in the textfile using the **textFile** method off of the SparkContext we created. This method will read a text file from HDFS, a local file system (available on all
# nodes), or any Hadoop-supported file system URI, and return it as an RDD of Strings.

# In[7]:


textFile = sc.textFile('example.txt')


# Spark’s primary abstraction is a distributed collection of items called a Resilient Distributed Dataset (RDD). RDDs can be created from Hadoop InputFormats (such as HDFS files) or by transforming other RDDs. 
# 
# ### Actions
# 
# We have just created an RDD using the textFile method and can perform operations on this object, such as counting the rows.
# 
# RDDs have actions, which return values, and transformations, which return pointers to new RDDs. Let’s start with a few actions:

# In[8]:


textFile.count()


# In[10]:


textFile.first()


# ### Transformations
# 
# Now we can use transformations, for example the filter transformation will return a new RDD with a subset of items in the file. Let's create a sample transformation using the filter() method. This method (just like Python's own filter function) will only return elements that satisfy the condition. Let's try looking for lines that contain the word 'second'. In which case, there should only be one line that has that.

# In[11]:


secfind = textFile.filter(lambda line: 'second' in line)


# In[12]:


# RDD
secfind


# In[13]:


# Perform action on transformation
secfind.collect()


# In[14]:


# Perform action on transformation
secfind.count()


# Notice how the transformations won't display an output and won't be run until an action is called. In the next lecture: Advanced Spark and Python we will begin to see many more examples of this transformation and action relationship!
# 
# # Great Job!
