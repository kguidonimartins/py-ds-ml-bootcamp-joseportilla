# 08-plotly-and-cufflinks_01-plotly_and_cufflinks

Install needed libraries

```python
!conda install plotly
!conda install -c conda-forge python-cufflinks
```

Check the official site of `plotly`: https://plot.ly/

`cufflinks` make the link between `pandas` and `plotly`: https://github.com/santosjorge/cufflinks

## Initial config

Load libraries

```python
import pandas as pd
import numpy as np
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
%matplotlib inline
```

Load data

```python
df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())
df.head()
df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [32, 43, 50]})
df2.head()
```

## iplot

```python
df.iplot()
```

## scatter

```python
df.iplot(kind='scatter', x='A', y='B', mode='markers', size=20)
```

## barplot

```python
df2.iplot(kind='bar', x='Category', y='Values')
```

Generate barplot for aggregated data.

```python
df.sum().iplot(kind='bar')
```

## boxplot

```python
df.iplot(kind='box')
```

## surface

```python
df3 = pd.DataFrame(
    {
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 30, 20, 10],
    'z': [5, 4, 3, 2, 1]
    }
)
df3.iplot(kind='surface', colorscale='rdylbu')
```

## histogram

```python
df['A'].iplot(kind='hist', bins=100)
```

or for all the dataset

```python
df.iplot(kind='hist')
```

## spred

```python
df[['A', 'B']].iplot(kind='spread')
```

## bubble

```python
df.iplot(kind='bubble', x='A', y='B', size='C')
```

## scatter matrix

```python
df.scatter_matrix()
```
