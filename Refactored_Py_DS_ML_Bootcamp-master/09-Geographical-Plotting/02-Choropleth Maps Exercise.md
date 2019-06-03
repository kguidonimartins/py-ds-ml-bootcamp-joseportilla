
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___

# Choropleth Maps Exercise 

Welcome to the Choropleth Maps Exercise! In this exercise we will give you some simple datasets and ask you to create Choropleth Maps from them. Due to the Nature of Plotly we can't show you examples

[Full Documentation Reference](https://plot.ly/python/reference/#choropleth)

## Plotly Imports


```python
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 
```

** Import pandas and read the csv file: 2014_World_Power_Consumption**


```python

```


```python

```

** Check the head of the DataFrame. **


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Power Consumption KWH</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>China</td>
      <td>5.523000e+12</td>
      <td>China 5,523,000,000,000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>3.832000e+12</td>
      <td>United 3,832,000,000,000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>European</td>
      <td>2.771000e+12</td>
      <td>European 2,771,000,000,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Russia</td>
      <td>1.065000e+12</td>
      <td>Russia 1,065,000,000,000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Japan</td>
      <td>9.210000e+11</td>
      <td>Japan 921,000,000,000</td>
    </tr>
  </tbody>
</table>
</div>



** Referencing the lecture notes, create a Choropleth Plot of the Power Consumption for Countries using the data and layout dictionary. **


```python

```


```python
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
```

## USA Choropleth

** Import the 2012_Election_Data csv file using pandas. **


```python

```

** Check the head of the DataFrame. **


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>ICPSR State Code</th>
      <th>Alphanumeric State Code</th>
      <th>State</th>
      <th>VEP Total Ballots Counted</th>
      <th>VEP Highest Office</th>
      <th>VAP Highest Office</th>
      <th>Total Ballots Counted</th>
      <th>Highest Office</th>
      <th>Voting-Eligible Population (VEP)</th>
      <th>Voting-Age Population (VAP)</th>
      <th>% Non-citizen</th>
      <th>Prison</th>
      <th>Probation</th>
      <th>Parole</th>
      <th>Total Ineligible Felon</th>
      <th>State Abv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>41</td>
      <td>1</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>58.6%</td>
      <td>56.0%</td>
      <td>NaN</td>
      <td>2,074,338</td>
      <td>3,539,217</td>
      <td>3707440.0</td>
      <td>2.6%</td>
      <td>32,232</td>
      <td>57,993</td>
      <td>8,616</td>
      <td>71,584</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>81</td>
      <td>2</td>
      <td>Alaska</td>
      <td>58.9%</td>
      <td>58.7%</td>
      <td>55.3%</td>
      <td>301,694</td>
      <td>300,495</td>
      <td>511,792</td>
      <td>543763.0</td>
      <td>3.8%</td>
      <td>5,633</td>
      <td>7,173</td>
      <td>1,882</td>
      <td>11,317</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>61</td>
      <td>3</td>
      <td>Arizona</td>
      <td>53.0%</td>
      <td>52.6%</td>
      <td>46.5%</td>
      <td>2,323,579</td>
      <td>2,306,559</td>
      <td>4,387,900</td>
      <td>4959270.0</td>
      <td>9.9%</td>
      <td>35,188</td>
      <td>72,452</td>
      <td>7,460</td>
      <td>81,048</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>42</td>
      <td>4</td>
      <td>Arkansas</td>
      <td>51.1%</td>
      <td>50.7%</td>
      <td>47.7%</td>
      <td>1,078,548</td>
      <td>1,069,468</td>
      <td>2,109,847</td>
      <td>2242740.0</td>
      <td>3.5%</td>
      <td>14,471</td>
      <td>30,122</td>
      <td>23,372</td>
      <td>53,808</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>71</td>
      <td>5</td>
      <td>California</td>
      <td>55.7%</td>
      <td>55.1%</td>
      <td>45.1%</td>
      <td>13,202,158</td>
      <td>13,038,547</td>
      <td>23,681,837</td>
      <td>28913129.0</td>
      <td>17.4%</td>
      <td>119,455</td>
      <td>0</td>
      <td>89,287</td>
      <td>208,742</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>



** Now create a plot that displays the Voting-Age Population (VAP) per state. If you later want to play around with other columns, make sure you consider their data type. VAP has already been transformed to a float for you. **


```python

```


```python

```


```python
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
```

# Great Job!
