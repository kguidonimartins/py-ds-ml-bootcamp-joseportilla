
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Pandas Data Visualization Exercise

This is just a quick exercise for you to review the various plots we showed earlier. Use **df3** to replicate the following plots. 


```python
import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('df3')
%matplotlib inline
```


```python
df3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 4 columns):
    a    500 non-null float64
    b    500 non-null float64
    c    500 non-null float64
    d    500 non-null float64
    dtypes: float64(4)
    memory usage: 15.7 KB



```python
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.336272</td>
      <td>0.325011</td>
      <td>0.001020</td>
      <td>0.401402</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.980265</td>
      <td>0.831835</td>
      <td>0.772288</td>
      <td>0.076485</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.480387</td>
      <td>0.686839</td>
      <td>0.000575</td>
      <td>0.746758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.502106</td>
      <td>0.305142</td>
      <td>0.768608</td>
      <td>0.654685</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.856602</td>
      <td>0.171448</td>
      <td>0.157971</td>
      <td>0.321231</td>
    </tr>
  </tbody>
</table>
</div>



**Recreate this scatter plot of b vs a. Note the color and size of the points. Also note the figure size. See if you can figure out how to stretch it in a similar fashion. Remeber back to your matplotlib lecture...**


```python
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

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f20142ac780>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_5_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1176a7da0>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_6_1.png)


**Create a histogram of the 'a' column.**


```python
df3.hist(column='a')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f1ff93a6358>]],
          dtype=object)




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_8_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1177a2860>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_9_1.png)


**These plots are okay, but they don't look very polished. Use style sheets to set the style to 'ggplot' and redo the histogram from above. Also figure out how to add more bins to it.**


```python
plt.style.use('ggplot')
```


```python
df3.hist(column='a', alpha=0.3, bins=30)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f1ff9435048>]],
          dtype=object)




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_12_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11a87b908>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_13_1.png)


**Create a boxplot comparing the a and b columns.**


```python
df3[['a', 'b']].boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1ff8a90e80>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_15_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1177c4a20>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_16_1.png)


**Create a kde plot of the 'd' column.**


```python
df3['d'].plot(kind='kde')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1ff701aba8>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_18_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11abb6278>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_19_1.png)


**Figure out how to increase the linewidth and make the linestyle dashed. (Note: You would usually not dash a kde plot line)**


```python
df3['d'].plot.density(
    linewidth=5, 
    linestyle='dashed'
)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2008d9ec18>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_21_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11ab9acc0>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_22_1.png)


**Create an area plot of all the columns for just the rows up to 30. (hint: use .ix).**


```python
df3.loc[0:30].plot.area(alpha=0.4)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1ff5647ba8>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_24_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11ccdfbe0>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_25_1.png)


## Bonus Challenge!
Note, you may find this really hard, reference the solutions if you can't figure it out!

**Notice how the legend in our previous figure overlapped some of actual diagram. Can you figure out how to display the legend outside of the plot as shown below?**

**Try searching Google for a good stackoverflow link on this topic. If you can't find it on your own - [use this one for a hint.](http://stackoverflow.com/questions/23556153/how-to-put-legend-outside-the-plot-with-pandas)**


```python
df3.loc[0:30].plot.area(alpha=0.4).legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ref: https://stackoverflow.com/a/4701285/5974372
```




    <matplotlib.legend.Legend at 0x7f1ff4d481d0>




![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_27_1.png)



```python

```


![png](02-Pandas%20Data%20Visualization%20Exercise_files/02-Pandas%20Data%20Visualization%20Exercise_28_0.png)


# Great Job!
