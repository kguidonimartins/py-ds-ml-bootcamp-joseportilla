# 09-geographical-plotting_01-choropleth_maps

Link to documentation: https://plot.ly/python/reference/#choropleth

```python
import pandas as pd
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
%matplotlib inline
```

## Create data

```python
data = dict(
    type = "choropleth",
    locations = ['AZ', 'CA', "NY"],
    locationmode = 'USA-states',
    colorscale = 'Portland',
    text = ['text 1', 'text 2', 'text 3'],
    z = [1.0, 2.0, 3.0],
    colorbar = {
        'title': 'Colorbar Title Goes Here'
        }
        )

layout = dict(geo = {'scope': 'usa'})
```

## Create figure

```python
choromap = go.Figure(data = [data], layout = layout)

iplot(choromap)
```

See in the browser

```python
plot(choromap)
```

## Load data

```python
df = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2011_US_AGRI_Exports')

df.head()
```

Create data and set layout

```python
data = dict(
    type = "choropleth",
    locations = df['code'],
    locationmode = 'USA-states',
    colorscale = 'Greens',
    text = df['text'],
    z = df['total exports'],
    colorbar = {'title': 'Millions USD'},
    marker = dict(line = dict(color = 'rgb(255, 255, 255)', width = 2))
    )

layout = dict(
    title = '2011 US Agriculture Export by State',
    geo = dict(
        scope = 'usa',
        showlakes = True,
        lakecolor = 'rgb(85, 173, 240)'
        )
    )
```

Create figure and plot

```python
choromap2 = go.Figure(data = [data], layout = layout)

iplot(choromap2)
```

## World map

Load data

```python
df = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2014_World_GDP')

df.head()
```

Prepare data and set layout

```python
data = dict(
    type = "choropleth",
    locations = df['CODE'],
    # colorscale = 'Greens',
    text = df['COUNTRY'],
    z = df['GDP (BILLIONS)'],
    colorbar = {'title': 'GDP in Billions USD'},
    # marker = dict(line = dict(color = 'rgb(255, 255, 255)', width = 2))
    )

layout = dict(
    title = '2014 Global GDP',
    geo = dict(
        showframe = False,
        # see at: https://plot.ly/python/reference/#layout-geo-projection
        projection = {'type': 'stereographic'}
        )
    )
```

Create figure and plot

```python
choromap3 = go.Figure(data = [data], layout = layout)

iplot(choromap3)
```
