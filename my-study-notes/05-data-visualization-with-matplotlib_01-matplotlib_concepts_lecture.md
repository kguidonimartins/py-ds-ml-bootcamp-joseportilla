# 05-data-visualization-with-matplotlib_01-matplotlib_concepts_lecture

Documentation: https://matplotlib.org/

```python

import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

```

Use `plt.show()` when you are not using Jupyter*, for example, when using a script like this:

```
#!/home/karlo/anaconda3/bin/python3

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 11)
y = x ** 2

plt.plot(x, y)
plt.show()
```

The script will remain running until you close the window that shows the figure. On the other hand, inside a Jupyter* works like executing a cell using 'string' or print('string').

## Create data

```python

x = np.linspace(0, 5, 11)
y = x ** 2

```

There are two methods to create plots using matplotlib: *i*) using functional programming and *ii*) using object oriented programming.

### Plot using functional programming

See: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

```python

plt.plot(x, y)

plt.show()

```

Changing the style: the sintaxe to change colors and markers is similar to MatLab.

```python

plt.plot(x, y, 'r-')
plt.plot(x, y, 'b-')
plt.plot(x, y, 'ro')
plt.plot(x, y, 'r+')
plt.plot(x, y, 'r--')
plt.plot(x, y, color = 'green', marker = 'o', linestyle = 'dashed', linewidth = 2, markersize = 12)

plt.show()

```

Adding basic labels

```python
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
```

Creating multiple plots

```python
plt.subplot(1, 2, 1)
plt.plot(x, y, 'r')

plt.subplot(1, 2, 2)
plt.plot(y, x, 'b')
```

### Plot using object oriented programming

```python
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot()
axes.set_xlabel("X Label")
axes.set_ylabel("Y Label")
axes.set_title("My title")
```

#### Creating an inset figure

```python
fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])

axes1.plot(x, y)
axes1.set_title("LARGER PLOT")
axes1.set_xlabel("X Label")
axes1.set_ylabel("Y Label")
axes2.plot(y, x)
axes2.set_title("SMALLER PLOT")
axes2.set_xlabel("X Label")
axes2.set_ylabel("Y Label")
```

#### Subplots

Plotting axes in a for loop.

```python
fig, axes = plt.subplots(nrows = 1, ncols = 2)
plt.tight_layout()

for current_ax in axes:
    current_ax.plot(x, y)
```

`axes` object is iterable! It can be plotted using their index.

```python
fig, axes = plt.subplots(nrows = 1, ncols = 2)
plt.tight_layout()

axes[0].plot(x, y)
axes[0].set_title("First plot")
axes[1].plot(y, x)
axes[1].set_title("Second plot")
```

### Changing the plot size

```python
fig = plt.figure(figsize=(8, 2))

ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y)
```

Works with subplot too.

```python
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(8,2))

axes[0].plot(x, y)
axes[1].plot(y, x)

plt.tight_layout()
```

### Legends

```python
fig = plt.figure()

ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y, label = "x, y")
ax.plot(y, x, label = "y, x")

ax.legend(loc=0)
```

### Saving figures

```python
fig

fig.savefig("my-study-notes/my_fig.png")
fig.savefig("my-study-notes/my_fig_200.png", dpi=200)
```

### Changing the plot appearance

```python
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(
    x,
    y,
    color='purple',
    # lw to linewidth works too
    linewidth=5,
    # ls to linestyle works too
    linestyle='-',
    alpha=1,
    marker='o',
    markersize=20,
    markerfacecolor='yellow',
    markeredgewidth=5,
    markeredgecolor='green'
    )
ax.set_xlim([0,1])
ax.set_ylim([0,2])
```

### Special plot types

- Scatter

```python
plt.scatter(x, y)
```

- Histogram

- Boxplot
