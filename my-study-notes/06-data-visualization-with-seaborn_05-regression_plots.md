# 06-data-visualization-with-seaborn_05-regression_plots

## Regression plots


```python
import seaborn as sns
%matplotlib inline
```

Load data

```python
tips = sns.load_dataset('tips')
tips.head()
```

### lm plot

```python
sns.lmplot(x='total_bill', y='tip', data=tips)
sns.lmplot(x='total_bill', y='tip', data=tips, hue='time')
sns.lmplot(x='total_bill', y='tip', data=tips, hue='time', markers=['o', 'v'])
```

You can adjust the other plot parameters using `matplotlib` keywords in a dictionary.

From the `seaborn`s documentation:

```
{scatter,line}_kws : dictionaries
Additional keyword arguments to pass to `plt.scatter` and `plt.plot`.
```

```python
sns.lmplot(
    x='total_bill',
    y='tip',
    data=tips,
    hue='time',
    markers=['o', 'v'],
    scatter_kws={
        's': 300,
        'alpha': 0.5
        }
        )
```

#### facet grid

Just change the `hue` parameter to `col` or `row`, or use both together.

```python
sns.lmplot(
    x='total_bill',
    y='tip',
    data=tips,
    col='time',
    row='sex'
        )
```

Using `hue` parameter in facet grid.


```python
sns.lmplot(
    x='total_bill',
    y='tip',
    data=tips,
    col='day',
    row='time',
    hue='sex'
        )
sns.lmplot(
    x='total_bill',
    y='tip',
    data=tips,
    col='day',
    hue='sex'
        )
```

Changing the aspect ratio and size.

```python
sns.lmplot(
    x='total_bill',
    y='tip',
    data=tips,
    col='day',
    hue='sex',
    aspect=0.6,
    size=5
        )
```
