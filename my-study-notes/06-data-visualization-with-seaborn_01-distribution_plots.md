# 06-data-visualization-with-seaborn_01-distribution_plots

## Distribution plots

```python

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

```

Load data

```python

tips = sns.load_dataset('tips')
tips.head()

```

### Histogram

```python

sns.distplot(
    tips['total_bill'],
    bins=None,
    hist=True,
    kde=True,
    rug=False,
    fit=None,
    hist_kws=None,
    kde_kws=None,
    rug_kws=None,
    fit_kws=None,
    color=None,
    vertical=False,
    norm_hist=False,
    axlabel=None,
    label=None,
    ax=None
    )
    
plt.show()
    
```

### Joint plot

```python

sns.jointplot(x='total_bill', y='tip', data=tips, kind='scatter')

plt.show()

```

### Pair plot

```python

sns.pairplot(tips, hue='sex')

plt.show()

```

### Rug plot

```python

sns.rugplot(tips['total_bill'])

plt.show()

```
