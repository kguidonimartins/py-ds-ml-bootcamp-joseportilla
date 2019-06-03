
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___

# Groupby

The groupby method allows you to group rows of data together and call aggregate functions


```python
import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
```


```python
df = pd.DataFrame(data)
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOG</td>
      <td>Sam</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOG</td>
      <td>Charlie</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>Amy</td>
      <td>340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MSFT</td>
      <td>Vanessa</td>
      <td>124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FB</td>
      <td>Carl</td>
      <td>243</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FB</td>
      <td>Sarah</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>



** Now you can use the .groupby() method to group rows together based off of a column name. For instance let's group based off of Company. This will create a DataFrameGroupBy object:**


```python
df.groupby('Company')
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x113014128>



You can save this object as a new variable:


```python
by_comp = df.groupby("Company")
```

And then call aggregate methods off the object:


```python
by_comp.mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>296.5</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>160.0</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>232.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Company').mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>296.5</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>160.0</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>232.0</td>
    </tr>
  </tbody>
</table>
</div>



More examples of aggregate methods:


```python
by_comp.std()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>75.660426</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>56.568542</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>152.735065</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_comp.min()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>Carl</td>
      <td>243</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>Charlie</td>
      <td>120</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>Amy</td>
      <td>124</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_comp.max()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>Sarah</td>
      <td>350</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>Sam</td>
      <td>200</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>Vanessa</td>
      <td>340</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_comp.count()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_comp.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">FB</th>
      <th>count</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>296.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>75.660426</td>
    </tr>
    <tr>
      <th>min</th>
      <td>243.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>269.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>296.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>323.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>350.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">GOOG</th>
      <th>count</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>160.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>56.568542</td>
    </tr>
    <tr>
      <th>min</th>
      <td>120.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>140.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>160.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>200.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">MSFT</th>
      <th>count</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>232.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>152.735065</td>
    </tr>
    <tr>
      <th>min</th>
      <td>124.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>232.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>286.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>340.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_comp.describe().transpose()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Company</th>
      <th colspan="8" halign="left">FB</th>
      <th colspan="5" halign="left">GOOG</th>
      <th colspan="8" halign="left">MSFT</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sales</th>
      <td>2.0</td>
      <td>296.5</td>
      <td>75.660426</td>
      <td>243.0</td>
      <td>269.75</td>
      <td>296.5</td>
      <td>323.25</td>
      <td>350.0</td>
      <td>2.0</td>
      <td>160.0</td>
      <td>...</td>
      <td>180.0</td>
      <td>200.0</td>
      <td>2.0</td>
      <td>232.0</td>
      <td>152.735065</td>
      <td>124.0</td>
      <td>178.0</td>
      <td>232.0</td>
      <td>286.0</td>
      <td>340.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 24 columns</p>
</div>




```python
by_comp.describe().transpose()['GOOG']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sales</th>
      <td>2.0</td>
      <td>160.0</td>
      <td>56.568542</td>
      <td>120.0</td>
      <td>140.0</td>
      <td>160.0</td>
      <td>180.0</td>
      <td>200.0</td>
    </tr>
  </tbody>
</table>
</div>



# Great Job!
