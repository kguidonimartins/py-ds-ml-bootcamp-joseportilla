
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___

# Grids

Grids are general types of plots that allow you to map plot types to rows and columns of a grid, this helps you create similar plots separated by features.


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
iris = sns.load_dataset('iris')
```


```python
iris.head()

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



## PairGrid

Pairgrid is a subplot grid for plotting pairwise relationships in a dataset.


```python
# Just the Grid
sns.PairGrid(iris)
```




    <seaborn.axisgrid.PairGrid at 0x11e39bb38>




![png](04-Grids_files/04-Grids_6_1.png)



```python
# Then you map to the grid
g = sns.PairGrid(iris)
g.map(plt.scatter)
```




    <seaborn.axisgrid.PairGrid at 0x11f431208>




![png](04-Grids_files/04-Grids_7_1.png)



```python
# Map to upper,lower, and diagonal
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
```




    <seaborn.axisgrid.PairGrid at 0x121944128>




![png](04-Grids_files/04-Grids_8_1.png)


## pairplot

pairplot is a simpler version of PairGrid (you'll use quite often)


```python
sns.pairplot(iris)
```




    <seaborn.axisgrid.PairGrid at 0x12024b5c0>




![png](04-Grids_files/04-Grids_10_1.png)



```python
sns.pairplot(iris,hue='species',palette='rainbow')
```




    <seaborn.axisgrid.PairGrid at 0x12633f0f0>




![png](04-Grids_files/04-Grids_11_1.png)


## Facet Grid

FacetGrid is the general way to create grids of plots based off of a feature:


```python
tips = sns.load_dataset('tips')
```


```python
tips.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Just the Grid
g = sns.FacetGrid(tips, col="time", row="smoker")
```


![png](04-Grids_files/04-Grids_15_0.png)



```python
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")
```


![png](04-Grids_files/04-Grids_16_0.png)



```python
g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')
# Notice hwo the arguments come after plt.scatter call
g = g.map(plt.scatter, "total_bill", "tip").add_legend()
```


![png](04-Grids_files/04-Grids_17_0.png)


## JointGrid

JointGrid is the general version for jointplot() type grids, for a quick example:


```python
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
```


![png](04-Grids_files/04-Grids_19_0.png)



```python
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(sns.regplot, sns.distplot)
```

    /Users/marci/anaconda/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j



![png](04-Grids_files/04-Grids_20_1.png)


Reference the documentation as necessary for grid types, but most of the time you'll just use the easier plots discussed earlier.
# Great Job!
