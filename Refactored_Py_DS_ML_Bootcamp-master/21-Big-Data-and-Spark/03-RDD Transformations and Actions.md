
# RDD Transformations and Actions

In this lecture we will begin to delve deeper into using Spark and Python. Please view the video lecture for a full explanation.

## Important Terms

Let's quickly go over some important terms:

Term                   |Definition
----                   |-------
RDD                    |Resilient Distributed Dataset
Transformation         |Spark operation that produces an RDD
Action                 |Spark operation that produces a local object
Spark Job              |Sequence of transformations on data with a final action

## Creating an RDD

There are two common ways to create an RDD:

Method                      |Result
----------                               |-------
`sc.parallelize(array)`                  |Create RDD of elements of array (or list)
`sc.textFile(path/to/file)`                      |Create RDD of lines from file

## RDD Transformations

We can use transformations to create a set of instructions we want to preform on the RDD (before we call an action and actually execute them).

Transformation Example                          |Result
----------                               |-------
`filter(lambda x: x % 2 == 0)`           |Discard non-even elements
`map(lambda x: x * 2)`                   |Multiply each RDD element by `2`
`map(lambda x: x.split())`               |Split each string into words
`flatMap(lambda x: x.split())`           |Split each string into words and flatten sequence
`sample(withReplacement=True,0.25)`      |Create sample of 25% of elements with replacement
`union(rdd)`                             |Append `rdd` to existing RDD
`distinct()`                             |Remove duplicates in RDD
`sortBy(lambda x: x, ascending=False)`   |Sort elements in descending order

## RDD Actions

Once you have your 'recipe' of transformations ready, what you will do next is execute them by calling an action. Here are some common actions:

Action                             |Result
----------                             |-------
`collect()`                            |Convert RDD to in-memory list 
`take(3)`                              |First 3 elements of RDD 
`top(3)`                               |Top 3 elements of RDD
`takeSample(withReplacement=True,3)`   |Create sample of 3 elements with replacement
`sum()`                                |Find element sum (assumes numeric elements)
`mean()`                               |Find element mean (assumes numeric elements)
`stdev()`                              |Find element deviation (assumes numeric elements)

____
# Examples

Now the best way to show all of this is by going through examples! We'll first review a bit by creating and working with a simple text file, then we will move on to more realistic data, such as customers and sales data.

### Creating an RDD from a text file:

** Creating the textfile **


```python
%%writefile example2.txt
first 
second line
the third line
then a fourth line
```

    Writing example2.txt


Now let's perform some transformations and actions on this text file:


```python
from pyspark import SparkContext
```


```python
sc = SparkContext()
```


```python
# Show RDD
sc.textFile('example2.txt')
```




    MapPartitionsRDD[1] at textFile at NativeMethodAccessorImpl.java:-2




```python
# Save a reference to this RDD
text_rdd = sc.textFile('example2.txt')
```


```python
# Map a function (or lambda expression) to each line
# Then collect the results.
text_rdd.map(lambda line: line.split()).collect()
```




    [['first'],
     ['second', 'line'],
     ['the', 'third', 'line'],
     ['then', 'a', 'fourth', 'line']]



## Map vs flatMap


```python
# Collect everything as a single flat map
text_rdd.flatMap(lambda line: line.split()).collect()
```




    ['first',
     'second',
     'line',
     'the',
     'third',
     'line',
     'then',
     'a',
     'fourth',
     'line']



# RDDs and Key Value Pairs

Now that we've worked with RDDs and how to aggregate values with them, we can begin to look into working with Key Value Pairs. In order to do this, let's create some fake data as a new text file.

This data represents some services sold to customers for some SAAS business.


```python
%%writefile services.txt
#EventId    Timestamp    Customer   State    ServiceID    Amount
201       10/13/2017      100       NY       131          100.00
204       10/18/2017      700       TX       129          450.00
202       10/15/2017      203       CA       121          200.00
206       10/19/2017      202       CA       131          500.00
203       10/17/2017      101       NY       173          750.00
205       10/19/2017      202       TX       121          200.00
```

    Writing services.txt



```python
services = sc.textFile('services.txt')
```


```python
services.take(2)
```




    ['#EventId    Timestamp    Customer   State    ServiceID    Amount',
     '201       10/13/2017      100       NY       131          100.00']




```python
services.map(lambda x: x.split())
```




    PythonRDD[10] at RDD at PythonRDD.scala:43




```python
services.map(lambda x: x.split()).take(3)
```




    [['#EventId', 'Timestamp', 'Customer', 'State', 'ServiceID', 'Amount'],
     ['201', '10/13/2017', '100', 'NY', '131', '100.00'],
     ['204', '10/18/2017', '700', 'TX', '129', '450.00']]



Let's remove that first hash-tag!


```python
services.map(lambda x: x[1:] if x[0]=='#' else x).collect()
```




    ['EventId    Timestamp    Customer   State    ServiceID    Amount',
     '201       10/13/2017      100       NY       131          100.00',
     '204       10/18/2017      700       TX       129          450.00',
     '202       10/15/2017      203       CA       121          200.00',
     '206       10/19/2017      202       CA       131          500.00',
     '203       10/17/2017      101       NY       173          750.00',
     '205       10/19/2017      202       TX       121          200.00']




```python
services.map(lambda x: x[1:] if x[0]=='#' else x).map(lambda x: x.split()).collect()
```




    [['EventId', 'Timestamp', 'Customer', 'State', 'ServiceID', 'Amount'],
     ['201', '10/13/2017', '100', 'NY', '131', '100.00'],
     ['204', '10/18/2017', '700', 'TX', '129', '450.00'],
     ['202', '10/15/2017', '203', 'CA', '121', '200.00'],
     ['206', '10/19/2017', '202', 'CA', '131', '500.00'],
     ['203', '10/17/2017', '101', 'NY', '173', '750.00'],
     ['205', '10/19/2017', '202', 'TX', '121', '200.00']]



## Using Key Value Pairs for Operations

Let us now begin to use methods that combine lambda expressions that use a ByKey argument. These ByKey methods will assume that your data is in a Key,Value form. 


For example let's find out the total sales per state: 


```python
# From Previous
cleanServ = services.map(lambda x: x[1:] if x[0]=='#' else x).map(lambda x: x.split())
```


```python
cleanServ.collect()
```




    [['EventId', 'Timestamp', 'Customer', 'State', 'ServiceID', 'Amount'],
     ['201', '10/13/2017', '100', 'NY', '131', '100.00'],
     ['204', '10/18/2017', '700', 'TX', '129', '450.00'],
     ['202', '10/15/2017', '203', 'CA', '121', '200.00'],
     ['206', '10/19/2017', '202', 'CA', '131', '500.00'],
     ['203', '10/17/2017', '101', 'NY', '173', '750.00'],
     ['205', '10/19/2017', '202', 'TX', '121', '200.00']]




```python
# Let's start by practicing grabbing fields
cleanServ.map(lambda lst: (lst[3],lst[-1])).collect()
```




    [('State', 'Amount'),
     ('NY', '100.00'),
     ('TX', '450.00'),
     ('CA', '200.00'),
     ('CA', '500.00'),
     ('NY', '750.00'),
     ('TX', '200.00')]




```python
# Continue with reduceByKey
# Notice how it assumes that the first item is the key!
cleanServ.map(lambda lst: (lst[3],lst[-1]))\
         .reduceByKey(lambda amt1,amt2 : amt1+amt2)\
         .collect()
```




    [('State', 'Amount'),
     ('NY', '100.00750.00'),
     ('TX', '450.00200.00'),
     ('CA', '200.00500.00')]



Uh oh! Looks like we forgot that the amounts are still strings! Let's fix that:


```python
# Continue with reduceByKey
# Notice how it assumes that the first item is the key!
cleanServ.map(lambda lst: (lst[3],lst[-1]))\
         .reduceByKey(lambda amt1,amt2 : float(amt1)+float(amt2))\
         .collect()
```




    [('State', 'Amount'), ('NY', 850.0), ('TX', 650.0), ('CA', 700.0)]



We can continue our analysis by sorting this output:


```python
# Grab state and amounts
# Add them
# Get rid of ('State','Amount')
# Sort them by the amount value
cleanServ.map(lambda lst: (lst[3],lst[-1]))\
.reduceByKey(lambda amt1,amt2 : float(amt1)+float(amt2))\
.filter(lambda x: not x[0]=='State')\
.sortBy(lambda stateAmount: stateAmount[1], ascending=False)\
.collect()
```




    [('NY', 850.0), ('CA', 700.0), ('TX', 650.0)]



** Remember to try to use unpacking for readability. For example: **


```python
x = ['ID','State','Amount']
```


```python
def func1(lst):
    return lst[-1]
```


```python
def func2(id_st_amt):
    # Unpack Values
    (Id,st,amt) = id_st_amt
    return amt
```


```python
func1(x)
```




    'Amount'




```python
func2(x)
```




    'Amount'



# Great Job!
