# 18-principal-component-analysis_01-principal_component_analysis

NOTE: See section 10.2 of ISLR book.

```python

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

```

Load data

```python

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()

```

Scale data

```python

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

```

PCA

```python

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape

```

Plot PCA

```python

plt.figure(figsize=(10, 8))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

plt.show()

```

Explained variance

```python

pca.explained_variance_ratio_

```
