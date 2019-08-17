# 06-data-visualization-with-seaborn_03-matrix_plots

## Matrix plots

```python
import seaborn as sns
%matplotlib inline
```

Load data

```python
tips = sns.load_dataset('tips')
tips.head()
flights = sns.load_dataset('flights')
flights.head()
```

### Heatmap

```python
tips_cor = tips.corr()
sns.heatmap(tips_cor)
sns.heatmap(tips_cor, annot=True)
sns.heatmap(tips_cor, annot=True, cmap='coolwarm')
```

Another example

```python
flights.head()
flights_pivot = flights.pivot_table(index='month', columns='year', values='passengers')
sns.heatmap(flights_pivot, cmap='magma')
sns.heatmap(flights_pivot, cmap='magma', linecolor='white', linewidths=3)
```

### Cluster map

```python
sns.clustermap(flights_pivot, cmap='coolwarm')
sns.clustermap(flights_pivot, cmap='coolwarm', standard_scale=1)
```
