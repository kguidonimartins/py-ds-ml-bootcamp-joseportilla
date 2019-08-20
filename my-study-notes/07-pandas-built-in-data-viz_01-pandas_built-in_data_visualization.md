# 07-pandas-built-in-data-viz_01-pandas_built-in_data_visualization

`pandas` dataviz built on top of `matplotlib`

```python
import numpy as np
import seaborn as sns
import pandas as pd
%matplotlib inline
```

Load data

```python
df1 = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df1', index_col=0)
df1.head()

df2 = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df2')
df2.head()
```

## Data viz

There are different methods to create plots with pandas.

```python
df1['A'].hist(bins=30)
df1['A'].plot(kind='hist', bins=30)
df1['A'].plot.hist(bins=30)
```

## Area plot

```python
df2.plot.area(alpha=0.6)
```

## Bar plot

```python
df2.plot.bar()
```

Stacked version

```python
df2.plot.bar(stacked=True)
```

## Histogram

```python
df1['A'].plot.hist(bins=30)
```

## Line plot

```python
df1.plot.line(y='B', figsize=(12, 3), lw=1)
```

## Scatter plot

```python
df1.plot.scatter(x='A', y='B')
```

Coloring based a third column

```python
df1.plot.scatter(
    x='A',
    y='B',
    c='C',
    s=df1['D']*100,
    cmap='viridis',
    sharex=False,
    alpha=0.5
    )
```

## Box plot

```python
df2.plot.box()
```

## Hex bin plot

```python
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df.plot.hexbin(x='a', y='b', gridsize=10, cmap='viridis')
```

## Kernel density estimation plot

```python
df2['a'].plot.kde()
df2['a'].plot.density()
df2.plot.density()
```
