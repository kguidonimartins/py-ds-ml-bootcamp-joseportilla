# 17-k-means-clustering_01-k_means_clustering_with_python

NOTE: See chapter 10 of ISLR book.

## Load libraries

```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
```

## Create data

```python
data = make_blobs(
    n_samples=200,
    n_features=2,
    centers=4,
    cluster_std=1.8,
    random_state=101
    )
```

```python
data[0]
data[1]
plt.scatter(data[0][:, 0], data[0][:, 1])
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
```

## Group KMeans with k = 4

```python
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
```

### Comparing original and predicted data

```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='viridis')
ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='viridis')
```

## Group KMeans with k = 2

```python
kmeans = KMeans(n_clusters=2)
kmeans.fit(data[0])
```

### Comparing original and predicted data

```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='viridis')
ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='viridis')
```
