
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___

# Style and Color

We've shown a few times how to control figure aesthetics in seaborn, but let's now go over it formally:


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
tips = sns.load_dataset('tips')
```

## Styles

You can set particular styles:


```python
sns.countplot(x='sex',data=tips)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11990cc88>




![png](06-Style%20and%20Color_files/06-Style%20and%20Color_4_1.png)



```python
sns.set_style('white')
sns.countplot(x='sex',data=tips)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c2ba9b0>




![png](06-Style%20and%20Color_files/06-Style%20and%20Color_5_1.png)



```python
sns.set_style('ticks')
sns.countplot(x='sex',data=tips,palette='deep')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119986978>




![png](06-Style%20and%20Color_files/06-Style%20and%20Color_6_1.png)


## Spine Removal


```python
sns.countplot(x='sex',data=tips)
sns.despine()
```


![png](06-Style%20and%20Color_files/06-Style%20and%20Color_8_0.png)



```python
sns.countplot(x='sex',data=tips)
sns.despine(left=True)
```


![png](06-Style%20and%20Color_files/06-Style%20and%20Color_9_0.png)


## Size and Aspect

You can use matplotlib's **plt.figure(figsize=(width,height) ** to change the size of most seaborn plots.

You can control the size and aspect ratio of most seaborn grid plots by passing in parameters: size, and aspect. For example:


```python
# Non Grid Plot
plt.figure(figsize=(12,3))
sns.countplot(x='sex',data=tips)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11cabbf28>




![png](06-Style%20and%20Color_files/06-Style%20and%20Color_12_1.png)



```python
# Grid Type Plot
sns.lmplot(x='total_bill',y='tip',size=2,aspect=4,data=tips)
```




    <seaborn.axisgrid.FacetGrid at 0x11cd69048>




![png](06-Style%20and%20Color_files/06-Style%20and%20Color_13_1.png)


## Scale and Context

The set_context() allows you to override default parameters:


```python
sns.set_context('poster',font_scale=4)
sns.countplot(x='sex',data=tips,palette='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11e2a2128>




![png](06-Style%20and%20Color_files/06-Style%20and%20Color_15_1.png)


Check out the documentation page for more info on these topics:
https://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html


```python
sns.puppyplot()
```

    /Users/marci/anaconda/lib/python3.5/site-packages/bs4/__init__.py:166: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    To get rid of this warning, change this:
    
     BeautifulSoup([your markup])
    
    to this:
    
     BeautifulSoup([your markup], "lxml")
    
      markup_type=markup_type))





<img alt="Titan the Pit Bull Terrier Pictures 1058495" class="" id="feature_image" src="http://cdn-www.dailypuppy.com/dog-images/titan-the-pit-bull-terrier_60497_2016-08-07_w450.jpg" style="width:450px;"/>



# Great Job!
