# 06-data-visualization-with-seaborn_02-categorical_plots

## Categorical plots

```python
import seaborn as sns
import numpy as np
%matplotlib inline
```

Load data

```python
tips = sns.load_dataset('tips')
tips.head()
```

### Bar plot

Categorical variables aggregated by numnpy functions or whatever aggregation function. The example below uses the standard deviation.

```python
sns.barplot(x='sex',y='total_bill', data=tips, estimator=np.std)
```

### Count plot

```python
sns.countplot(x='sex', data=tips)
```

### Box plot

```python
sns.boxplot(x='day', y='total_bill', data=tips)
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
```

### Violin plot

```python
sns.violinplot(x='day', y='total_bill', data=tips)
sns.violinplot(x='day', y='total_bill', data=tips, hue='smoker')
sns.violinplot(x='day', y='total_bill', data=tips, hue='smoker', split=True)
```

### Strip plot

```python
sns.stripplot(x='day', y='total_bill', data=tips, hue='sex', split='True')
sns.stripplot(x='day', y='total_bill', data=tips, jitter=False)
```

### Swarn plot

```python
sns.swarmplot(x='day', y='total_bill', data=tips)
```

Combining with violin plot

```python
sns.violinplot(x='day', y='total_bill', data=tips)
sns.swarmplot(x='day', y='total_bill', data=tips, color='black')
```

### Factor plot

```python
sns.factorplot(x='day', y='total_bill', data=tips, kind='bar')
sns.factorplot(x='day', y='total_bill', data=tips, kind='violin')
```
