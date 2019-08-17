# 06-data-visualization-with-seaborn_04-grids

## Grids plots

```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

Load data

```python
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
iris.head()
iris['species'].unique()
```

### Pair grid plot

```python
sns.pairplot(iris)
g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(sns.kdeplot)
g.map_lower(plt.scatter)
```

Another example

```python
g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.distplot, 'total_bill')
```

Faceting with matplotlib

```python
g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(plt.scatter, 'total_bill', 'tip')
```
