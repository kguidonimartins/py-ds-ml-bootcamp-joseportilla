# 06-data-visualization-with-seaborn_06-style_and_color

## Style and color

Load libs

```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

Load data

```python
tips = sns.load_dataset('tips')
tips.head()
```

### Setting style

Set background lines

```python
sns.set_style('whitegrid')
sns.countplot(x='sex', data=tips)
```

Set boxes and ticks

```python
sns.set_style('ticks')
sns.countplot(x='sex', data=tips)
sns.despine()
```

You can specify the parameters to remove.

```python
sns.set_style('ticks')
sns.countplot(x='sex', data=tips)
sns.despine(left=True, bottom=True, right=True, top=True)
```

Changing the context of the figure (*a.k.a.*, changing aspect ratio and scale)

```python
sns.set_context(context='poster')
sns.countplot(x='sex', data=tips)
```

Changing the font scale

```python
sns.set_context(context='poster', font_scale=2)
sns.countplot(x='sex', data=tips)
```

Setting palette and colors.

Find the string colors in `matplotlib` documentation for [`colormap`](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)

```python
sns.set_context(context='notebook', font_scale=2)
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='seismic', scatter_kws={'alpha': 0.5})
```

### Setting style with `matplotlib` parameters

```python
plt.figure(figsize=(12, 3))
sns.countplot(x='sex', data=tips)
```
