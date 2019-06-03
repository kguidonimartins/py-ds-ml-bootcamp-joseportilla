
# Introduction to Spark and Python

Let's learn how to use Spark with Python by using the pyspark library! Make sure to view the video lecture explaining Spark and RDDs before continuing on with this code.

This notebook will serve as reference code for the Big Data section of the course involving Amazon Web Services. The video will provide fuller explanations for what the code is doing.

## Creating a SparkContext

First we need to create a SparkContext. We will import this from pyspark:


```python
from pyspark import SparkContext
```

Now create the SparkContext,A SparkContext represents the connection to a Spark cluster, and can be used to create an RDD and broadcast variables on that cluster.

*Note! You can only have one SparkContext at a time the way we are running things here.*


```python
sc = SparkContext()
```

## Basic Operations

We're going to start with a 'hello world' example, which is just reading a text file. First let's create a text file.
___

Let's write an example text file to read, we'll use some special jupyter notebook commands for this, but feel free to use any .txt file:


```python
%%writefile example.txt
first line
second line
third line
fourth line
```

    Overwriting example.txt


### Creating the RDD

Now we can take in the textfile using the **textFile** method off of the SparkContext we created. This method will read a text file from HDFS, a local file system (available on all
nodes), or any Hadoop-supported file system URI, and return it as an RDD of Strings.


```python
textFile = sc.textFile('example.txt')
```

Spark’s primary abstraction is a distributed collection of items called a Resilient Distributed Dataset (RDD). RDDs can be created from Hadoop InputFormats (such as HDFS files) or by transforming other RDDs. 

### Actions

We have just created an RDD using the textFile method and can perform operations on this object, such as counting the rows.

RDDs have actions, which return values, and transformations, which return pointers to new RDDs. Let’s start with a few actions:


```python
textFile.count()
```




    4




```python
textFile.first()
```




    'first line'



### Transformations

Now we can use transformations, for example the filter transformation will return a new RDD with a subset of items in the file. Let's create a sample transformation using the filter() method. This method (just like Python's own filter function) will only return elements that satisfy the condition. Let's try looking for lines that contain the word 'second'. In which case, there should only be one line that has that.


```python
secfind = textFile.filter(lambda line: 'second' in line)
```


```python
# RDD
secfind
```




    PythonRDD[7] at RDD at PythonRDD.scala:43




```python
# Perform action on transformation
secfind.collect()
```




    ['second line']




```python
# Perform action on transformation
secfind.count()
```




    1



Notice how the transformations won't display an output and won't be run until an action is called. In the next lecture: Advanced Spark and Python we will begin to see many more examples of this transformation and action relationship!

# Great Job!
