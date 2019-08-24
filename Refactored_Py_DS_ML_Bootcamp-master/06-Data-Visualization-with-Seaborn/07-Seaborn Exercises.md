
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Seaborn Exercises

Time to practice your new seaborn skills! Try to recreate the plots below (don't worry about color schemes, just the plot itself.

## The Data

We will be working with a famous titanic data set for these exercises. Later on in the Machine Learning section of the course, we will revisit this data, and use it to predict survival rates of passengers. For now, we'll just focus on the visualization of the data with seaborn:


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
sns.set_style('whitegrid')
```


```python
titanic = sns.load_dataset('titanic')
```


```python
titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



# Exercises

**Recreate the plots below using the titanic dataframe. There are very few hints since most of the plots can be done with just one or two lines of code and a hint would basically give away the solution. Keep careful attention to the x and y labels for hints.**

**Note! In order to not lose the plot image, make sure you don't code in the cell that is directly above the plot, there is an extra cell above that one which won't overwrite that plot!**


```python
# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
```


```python
?sns.jointplot
```


    [0;31mSignature:[0m
    [0msns[0m[0;34m.[0m[0mjointplot[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mx[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0my[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdata[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkind[0m[0;34m=[0m[0;34m'scatter'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mstat_func[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcolor[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mheight[0m[0;34m=[0m[0;36m6[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mratio[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mspace[0m[0;34m=[0m[0;36m0.2[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdropna[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mxlim[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mylim[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mjoint_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmarginal_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mannot_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Draw a plot of two variables with bivariate and univariate graphs.
    
    This function provides a convenient interface to the :class:`JointGrid`
    class, with several canned plot kinds. This is intended to be a fairly
    lightweight wrapper; if you need more flexibility, you should use
    :class:`JointGrid` directly.
    
    Parameters
    ----------
    x, y : strings or vectors
        Data or names of variables in ``data``.
    data : DataFrame, optional
        DataFrame when ``x`` and ``y`` are variable names.
    kind : { "scatter" | "reg" | "resid" | "kde" | "hex" }, optional
        Kind of plot to draw.
    stat_func : callable or None, optional
        *Deprecated*
    color : matplotlib color, optional
        Color used for the plot elements.
    height : numeric, optional
        Size of the figure (it will be square).
    ratio : numeric, optional
        Ratio of joint axes height to marginal axes height.
    space : numeric, optional
        Space between the joint and marginal axes
    dropna : bool, optional
        If True, remove observations that are missing from ``x`` and ``y``.
    {x, y}lim : two-tuples, optional
        Axis limits to set before plotting.
    {joint, marginal, annot}_kws : dicts, optional
        Additional keyword arguments for the plot components.
    kwargs : key, value pairings
        Additional keyword arguments are passed to the function used to
        draw the plot on the joint Axes, superseding items in the
        ``joint_kws`` dictionary.
    
    Returns
    -------
    grid : :class:`JointGrid`
        :class:`JointGrid` object with the plot on it.
    
    See Also
    --------
    JointGrid : The Grid class used for drawing this plot. Use it directly if
                you need more flexibility.
    
    Examples
    --------
    
    Draw a scatterplot with marginal histograms:
    
    .. plot::
        :context: close-figs
    
        >>> import numpy as np, pandas as pd; np.random.seed(0)
        >>> import seaborn as sns; sns.set(style="white", color_codes=True)
        >>> tips = sns.load_dataset("tips")
        >>> g = sns.jointplot(x="total_bill", y="tip", data=tips)
    
    Add regression and kernel density fits:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.jointplot("total_bill", "tip", data=tips, kind="reg")
    
    Replace the scatterplot with a joint histogram using hexagonal bins:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.jointplot("total_bill", "tip", data=tips, kind="hex")
    
    Replace the scatterplots and histograms with density estimates and align
    the marginal Axes tightly with the joint Axes:
    
    .. plot::
        :context: close-figs
    
        >>> iris = sns.load_dataset("iris")
        >>> g = sns.jointplot("sepal_width", "petal_length", data=iris,
        ...                   kind="kde", space=0, color="g")
    
    Draw a scatterplot, then add a joint density estimate:
    
    .. plot::
        :context: close-figs
    
        >>> g = (sns.jointplot("sepal_length", "sepal_width",
        ...                    data=iris, color="k")
        ...         .plot_joint(sns.kdeplot, zorder=0, n_levels=6))
    
    Pass vectors in directly without using Pandas, then name the axes:
    
    .. plot::
        :context: close-figs
    
        >>> x, y = np.random.randn(2, 300)
        >>> g = (sns.jointplot(x, y, kind="hex")
        ...         .set_axis_labels("x", "y"))
    
    Draw a smaller figure with more space devoted to the marginal plots:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.jointplot("total_bill", "tip", data=tips,
        ...                   height=5, ratio=3, color="g")
    
    Pass keyword arguments down to the underlying plots:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.jointplot("petal_length", "sepal_length", data=iris,
        ...                   marginal_kws=dict(bins=15, rug=True),
        ...                   annot_kws=dict(stat="r"),
        ...                   s=40, edgecolor="w", linewidth=1)
    [0;31mFile:[0m      ~/anaconda3/lib/python3.7/site-packages/seaborn/axisgrid.py
    [0;31mType:[0m      function




```python
sns.jointplot(
    x='fare', 
    y='age', 
    data=titanic, 
    kind='scatter'
)
```




    <seaborn.axisgrid.JointGrid at 0x7ff9a7103358>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_9_1.png)



```python

```




    <seaborn.axisgrid.JointGrid at 0x11d0389e8>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_10_1.png)



```python
# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
```


```python
?sns.distplot
```


    [0;31mSignature:[0m
    [0msns[0m[0;34m.[0m[0mdistplot[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0ma[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mbins[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhist[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkde[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrug[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mfit[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhist_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkde_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrug_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mfit_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcolor[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mvertical[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mnorm_hist[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0maxlabel[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlabel[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0max[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Flexibly plot a univariate distribution of observations.
    
    This function combines the matplotlib ``hist`` function (with automatic
    calculation of a good default bin size) with the seaborn :func:`kdeplot`
    and :func:`rugplot` functions. It can also fit ``scipy.stats``
    distributions and plot the estimated PDF over the data.
    
    Parameters
    ----------
    
    a : Series, 1d-array, or list.
        Observed data. If this is a Series object with a ``name`` attribute,
        the name will be used to label the data axis.
    bins : argument for matplotlib hist(), or None, optional
        Specification of hist bins, or None to use Freedman-Diaconis rule.
    hist : bool, optional
        Whether to plot a (normed) histogram.
    kde : bool, optional
        Whether to plot a gaussian kernel density estimate.
    rug : bool, optional
        Whether to draw a rugplot on the support axis.
    fit : random variable object, optional
        An object with `fit` method, returning a tuple that can be passed to a
        `pdf` method a positional arguments following an grid of values to
        evaluate the pdf on.
    {hist, kde, rug, fit}_kws : dictionaries, optional
        Keyword arguments for underlying plotting functions.
    color : matplotlib color, optional
        Color to plot everything but the fitted curve in.
    vertical : bool, optional
        If True, observed values are on y-axis.
    norm_hist : bool, optional
        If True, the histogram height shows a density rather than a count.
        This is implied if a KDE or fitted density is plotted.
    axlabel : string, False, or None, optional
        Name for the support axis label. If None, will try to get it
        from a.namel if False, do not set a label.
    label : string, optional
        Legend label for the relevent component of the plot
    ax : matplotlib axis, optional
        if provided, plot on this axis
    
    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    
    See Also
    --------
    kdeplot : Show a univariate or bivariate distribution with a kernel
              density estimate.
    rugplot : Draw small vertical lines to show each observation in a
              distribution.
    
    Examples
    --------
    
    Show a default plot with a kernel density estimate and histogram with bin
    size determined automatically with a reference rule:
    
    .. plot::
        :context: close-figs
    
        >>> import seaborn as sns, numpy as np
        >>> sns.set(); np.random.seed(0)
        >>> x = np.random.randn(100)
        >>> ax = sns.distplot(x)
    
    Use Pandas objects to get an informative axis label:
    
    .. plot::
        :context: close-figs
    
        >>> import pandas as pd
        >>> x = pd.Series(x, name="x variable")
        >>> ax = sns.distplot(x)
    
    Plot the distribution with a kernel density estimate and rug plot:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.distplot(x, rug=True, hist=False)
    
    Plot the distribution with a histogram and maximum likelihood gaussian
    distribution fit:
    
    .. plot::
        :context: close-figs
    
        >>> from scipy.stats import norm
        >>> ax = sns.distplot(x, fit=norm, kde=False)
    
    Plot the distribution on the vertical axis:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.distplot(x, vertical=True)
    
    Change the color of all the plot elements:
    
    .. plot::
        :context: close-figs
    
        >>> sns.set_color_codes()
        >>> ax = sns.distplot(x, color="y")
    
    Pass specific parameters to the underlying plot functions:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.distplot(x, rug=True, rug_kws={"color": "g"},
        ...                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
        ...                   hist_kws={"histtype": "step", "linewidth": 3,
        ...                             "alpha": 1, "color": "g"})
    [0;31mFile:[0m      ~/anaconda3/lib/python3.7/site-packages/seaborn/distributions.py
    [0;31mType:[0m      function




```python
sns.distplot(
    a=titanic['fare'],
    bins=30,
    kde=None
)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff9a5d01748>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_13_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11fc5ca90>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_14_1.png)



```python
# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
```


```python
?sns.boxplot
```


    [0;31mSignature:[0m
    [0msns[0m[0;34m.[0m[0mboxplot[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdata[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0morder[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue_order[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0morient[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcolor[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpalette[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msaturation[0m[0;34m=[0m[0;36m0.75[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mwidth[0m[0;34m=[0m[0;36m0.8[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdodge[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mfliersize[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlinewidth[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mwhis[0m[0;34m=[0m[0;36m1.5[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mnotch[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0max[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Draw a box plot to show distributions with respect to categories.
    
    A box plot (or box-and-whisker plot) shows the distribution of quantitative
    data in a way that facilitates comparisons between variables or across
    levels of a categorical variable. The box shows the quartiles of the
    dataset while the whiskers extend to show the rest of the distribution,
    except for points that are determined to be "outliers" using a method
    that is a function of the inter-quartile range.
    
    
    Input data can be passed in a variety of formats, including:
    
    - Vectors of data represented as lists, numpy arrays, or pandas Series
      objects passed directly to the ``x``, ``y``, and/or ``hue`` parameters.
    - A "long-form" DataFrame, in which case the ``x``, ``y``, and ``hue``
      variables will determine how the data are plotted.
    - A "wide-form" DataFrame, such that each numeric column will be plotted.
    - An array or list of vectors.
    
    In most cases, it is possible to use numpy or Python objects, but pandas
    objects are preferable because the associated names will be used to
    annotate the axes. Additionally, you can use Categorical types for the
    grouping variables to control the order of plot elements.    
    
    This function always treats one of the variables as categorical and
    draws data at ordinal positions (0, 1, ... n) on the relevant axis, even
    when the data has a numeric or date type.
    
    See the :ref:`tutorial <categorical_tutorial>` for more information.    
    
    Parameters
    ----------
    x, y, hue : names of variables in ``data`` or vector data, optional
        Inputs for plotting long-form data. See examples for interpretation.        
    data : DataFrame, array, or list of arrays, optional
        Dataset for plotting. If ``x`` and ``y`` are absent, this is
        interpreted as wide-form. Otherwise it is expected to be long-form.    
    order, hue_order : lists of strings, optional
        Order to plot the categorical levels in, otherwise the levels are
        inferred from the data objects.        
    orient : "v" | "h", optional
        Orientation of the plot (vertical or horizontal). This is usually
        inferred from the dtype of the input variables, but can be used to
        specify when the "categorical" variable is a numeric or when plotting
        wide-form data.    
    color : matplotlib color, optional
        Color for all of the elements, or seed for a gradient palette.    
    palette : palette name, list, or dict, optional
        Colors to use for the different levels of the ``hue`` variable. Should
        be something that can be interpreted by :func:`color_palette`, or a
        dictionary mapping hue levels to matplotlib colors.    
    saturation : float, optional
        Proportion of the original saturation to draw colors at. Large patches
        often look better with slightly desaturated colors, but set this to
        ``1`` if you want the plot colors to perfectly match the input color
        spec.    
    width : float, optional
        Width of a full element when not using hue nesting, or width of all the
        elements for one level of the major grouping variable.    
    dodge : bool, optional
        When hue nesting is used, whether elements should be shifted along the
        categorical axis.    
    fliersize : float, optional
        Size of the markers used to indicate outlier observations.
    linewidth : float, optional
        Width of the gray lines that frame the plot elements.    
    whis : float, optional
        Proportion of the IQR past the low and high quartiles to extend the
        plot whiskers. Points outside this range will be identified as
        outliers.
    notch : boolean, optional
        Whether to "notch" the box to indicate a confidence interval for the
        median. There are several other parameters that can control how the
        notches are drawn; see the ``plt.boxplot`` help for more information
        on them.
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.    
    kwargs : key, value mappings
        Other keyword arguments are passed through to ``plt.boxplot`` at draw
        time.
    
    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.    
    
    See Also
    --------
    violinplot : A combination of boxplot and kernel density estimation.    
    stripplot : A scatterplot where one variable is categorical. Can be used
                in conjunction with other plots to show each observation.    
    swarmplot : A categorical scatterplot where the points do not overlap. Can
                be used with other plots to show each observation.    
    
    Examples
    --------
    
    Draw a single horizontal boxplot:
    
    .. plot::
        :context: close-figs
    
        >>> import seaborn as sns
        >>> sns.set(style="whitegrid")
        >>> tips = sns.load_dataset("tips")
        >>> ax = sns.boxplot(x=tips["total_bill"])
    
    Draw a vertical boxplot grouped by a categorical variable:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.boxplot(x="day", y="total_bill", data=tips)
    
    Draw a boxplot with nested grouping by two categorical variables:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
        ...                  data=tips, palette="Set3")
    
    Draw a boxplot with nested grouping when some bins are empty:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.boxplot(x="day", y="total_bill", hue="time",
        ...                  data=tips, linewidth=2.5)
    
    Control box order by passing an explicit order:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.boxplot(x="time", y="tip", data=tips,
        ...                  order=["Dinner", "Lunch"])
    
    Draw a boxplot for each numeric variable in a DataFrame:
    
    .. plot::
        :context: close-figs
    
        >>> iris = sns.load_dataset("iris")
        >>> ax = sns.boxplot(data=iris, orient="h", palette="Set2")
    
    Use ``hue`` without changing box position or width:
    
    .. plot::
        :context: close-figs
    
        >>> tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
        >>> ax = sns.boxplot(x="day", y="total_bill", hue="weekend",
        ...                  data=tips, dodge=False)
    
    Use :func:`swarmplot` to show the datapoints on top of the boxes:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.boxplot(x="day", y="total_bill", data=tips)
        >>> ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
    
    Use :func:`catplot` to combine a :func:`pointplot` and a
    :class:`FacetGrid`. This allows grouping within additional categorical
    variables. Using :func:`catplot` is safer than using :class:`FacetGrid`
    directly, as it ensures synchronization of variable order across facets:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.catplot(x="sex", y="total_bill",
        ...                 hue="smoker", col="time",
        ...                 data=tips, kind="box",
        ...                 height=4, aspect=.7);
    [0;31mFile:[0m      ~/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py
    [0;31mType:[0m      function




```python
sns.boxplot(x='class', y='age', data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff9a5f89c50>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_17_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f23da90>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_18_1.png)



```python
# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
```


```python
?sns.swarmplot
```


    [0;31mSignature:[0m
    [0msns[0m[0;34m.[0m[0mswarmplot[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdata[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0morder[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue_order[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdodge[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0morient[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcolor[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpalette[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msize[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0medgecolor[0m[0;34m=[0m[0;34m'gray'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlinewidth[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0max[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Draw a categorical scatterplot with non-overlapping points.
    
    This function is similar to :func:`stripplot`, but the points are adjusted
    (only along the categorical axis) so that they don't overlap. This gives a
    better representation of the distribution of values, but it does not scale
    well to large numbers of observations. This style of plot is sometimes
    called a "beeswarm".
    
    A swarm plot can be drawn on its own, but it is also a good complement
    to a box or violin plot in cases where you want to show all observations
    along with some representation of the underlying distribution.
    
    Arranging the points properly requires an accurate transformation between
    data and point coordinates. This means that non-default axis limits must
    be set *before* drawing the plot.
    
    
    Input data can be passed in a variety of formats, including:
    
    - Vectors of data represented as lists, numpy arrays, or pandas Series
      objects passed directly to the ``x``, ``y``, and/or ``hue`` parameters.
    - A "long-form" DataFrame, in which case the ``x``, ``y``, and ``hue``
      variables will determine how the data are plotted.
    - A "wide-form" DataFrame, such that each numeric column will be plotted.
    - An array or list of vectors.
    
    In most cases, it is possible to use numpy or Python objects, but pandas
    objects are preferable because the associated names will be used to
    annotate the axes. Additionally, you can use Categorical types for the
    grouping variables to control the order of plot elements.    
    
    This function always treats one of the variables as categorical and
    draws data at ordinal positions (0, 1, ... n) on the relevant axis, even
    when the data has a numeric or date type.
    
    See the :ref:`tutorial <categorical_tutorial>` for more information.    
    
    Parameters
    ----------
    x, y, hue : names of variables in ``data`` or vector data, optional
        Inputs for plotting long-form data. See examples for interpretation.        
    data : DataFrame, array, or list of arrays, optional
        Dataset for plotting. If ``x`` and ``y`` are absent, this is
        interpreted as wide-form. Otherwise it is expected to be long-form.    
    order, hue_order : lists of strings, optional
        Order to plot the categorical levels in, otherwise the levels are
        inferred from the data objects.        
    dodge : bool, optional
        When using ``hue`` nesting, setting this to ``True`` will separate
        the strips for different hue levels along the categorical axis.
        Otherwise, the points for each level will be plotted in one swarm.
    orient : "v" | "h", optional
        Orientation of the plot (vertical or horizontal). This is usually
        inferred from the dtype of the input variables, but can be used to
        specify when the "categorical" variable is a numeric or when plotting
        wide-form data.    
    color : matplotlib color, optional
        Color for all of the elements, or seed for a gradient palette.    
    palette : palette name, list, or dict, optional
        Colors to use for the different levels of the ``hue`` variable. Should
        be something that can be interpreted by :func:`color_palette`, or a
        dictionary mapping hue levels to matplotlib colors.    
    size : float, optional
        Diameter of the markers, in points. (Although ``plt.scatter`` is used
        to draw the points, the ``size`` argument here takes a "normal"
        markersize and not size^2 like ``plt.scatter``.
    edgecolor : matplotlib color, "gray" is special-cased, optional
        Color of the lines around each point. If you pass ``"gray"``, the
        brightness is determined by the color palette used for the body
        of the points.
    linewidth : float, optional
        Width of the gray lines that frame the plot elements.    
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.    
    
    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.    
    
    See Also
    --------
    boxplot : A traditional box-and-whisker plot with a similar API.    
    violinplot : A combination of boxplot and kernel density estimation.    
    stripplot : A scatterplot where one variable is categorical. Can be used
                in conjunction with other plots to show each observation.    
    catplot : Combine a categorical plot with a class:`FacetGrid`.    
    
    Examples
    --------
    
    Draw a single horizontal swarm plot:
    
    .. plot::
        :context: close-figs
    
        >>> import seaborn as sns
        >>> sns.set(style="whitegrid")
        >>> tips = sns.load_dataset("tips")
        >>> ax = sns.swarmplot(x=tips["total_bill"])
    
    Group the swarms by a categorical variable:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.swarmplot(x="day", y="total_bill", data=tips)
    
    Draw horizontal swarms:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.swarmplot(x="total_bill", y="day", data=tips)
    
    Color the points using a second categorical variable:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)
    
    Split each level of the ``hue`` variable along the categorical axis:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.swarmplot(x="day", y="total_bill", hue="smoker",
        ...                    data=tips, palette="Set2", dodge=True)
    
    Control swarm order by passing an explicit order:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.swarmplot(x="time", y="tip", data=tips,
        ...                    order=["Dinner", "Lunch"])
    
    Plot using larger points:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.swarmplot(x="time", y="tip", data=tips, size=6)
    
    Draw swarms of observations on top of a box plot:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.boxplot(x="tip", y="day", data=tips, whis=np.inf)
        >>> ax = sns.swarmplot(x="tip", y="day", data=tips, color=".2")
    
    Draw swarms of observations on top of a violin plot:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
        >>> ax = sns.swarmplot(x="day", y="total_bill", data=tips,
        ...                    color="white", edgecolor="gray")
    
    Use :func:`catplot` to combine a :func:`swarmplot` and a
    :class:`FacetGrid`. This allows grouping within additional categorical
    variables. Using :func:`catplot` is safer than using :class:`FacetGrid`
    directly, as it ensures synchronization of variable order across facets:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.catplot(x="sex", y="total_bill",
        ...                 hue="smoker", col="time",
        ...                 data=tips, kind="swarm",
        ...                 height=4, aspect=.7);
    [0;31mFile:[0m      ~/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py
    [0;31mType:[0m      function




```python
sns.swarmplot(x='class', y='age', data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff9a5e2edd8>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_21_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f215320>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_22_1.png)



```python
# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
```


```python
?sns.countplot
```


    [0;31mSignature:[0m
    [0msns[0m[0;34m.[0m[0mcountplot[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdata[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0morder[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue_order[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0morient[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcolor[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpalette[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msaturation[0m[0;34m=[0m[0;36m0.75[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdodge[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0max[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Show the counts of observations in each categorical bin using bars.
    
    A count plot can be thought of as a histogram across a categorical, instead
    of quantitative, variable. The basic API and options are identical to those
    for :func:`barplot`, so you can compare counts across nested variables.
    
    
    Input data can be passed in a variety of formats, including:
    
    - Vectors of data represented as lists, numpy arrays, or pandas Series
      objects passed directly to the ``x``, ``y``, and/or ``hue`` parameters.
    - A "long-form" DataFrame, in which case the ``x``, ``y``, and ``hue``
      variables will determine how the data are plotted.
    - A "wide-form" DataFrame, such that each numeric column will be plotted.
    - An array or list of vectors.
    
    In most cases, it is possible to use numpy or Python objects, but pandas
    objects are preferable because the associated names will be used to
    annotate the axes. Additionally, you can use Categorical types for the
    grouping variables to control the order of plot elements.    
    
    This function always treats one of the variables as categorical and
    draws data at ordinal positions (0, 1, ... n) on the relevant axis, even
    when the data has a numeric or date type.
    
    See the :ref:`tutorial <categorical_tutorial>` for more information.    
    
    Parameters
    ----------
    x, y, hue : names of variables in ``data`` or vector data, optional
        Inputs for plotting long-form data. See examples for interpretation.        
    data : DataFrame, array, or list of arrays, optional
        Dataset for plotting. If ``x`` and ``y`` are absent, this is
        interpreted as wide-form. Otherwise it is expected to be long-form.    
    order, hue_order : lists of strings, optional
        Order to plot the categorical levels in, otherwise the levels are
        inferred from the data objects.        
    orient : "v" | "h", optional
        Orientation of the plot (vertical or horizontal). This is usually
        inferred from the dtype of the input variables, but can be used to
        specify when the "categorical" variable is a numeric or when plotting
        wide-form data.    
    color : matplotlib color, optional
        Color for all of the elements, or seed for a gradient palette.    
    palette : palette name, list, or dict, optional
        Colors to use for the different levels of the ``hue`` variable. Should
        be something that can be interpreted by :func:`color_palette`, or a
        dictionary mapping hue levels to matplotlib colors.    
    saturation : float, optional
        Proportion of the original saturation to draw colors at. Large patches
        often look better with slightly desaturated colors, but set this to
        ``1`` if you want the plot colors to perfectly match the input color
        spec.    
    dodge : bool, optional
        When hue nesting is used, whether elements should be shifted along the
        categorical axis.    
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.    
    kwargs : key, value mappings
        Other keyword arguments are passed to ``plt.bar``.
    
    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.    
    
    See Also
    --------
    barplot : Show point estimates and confidence intervals using bars.    
    catplot : Combine a categorical plot with a class:`FacetGrid`.    
    
    Examples
    --------
    
    Show value counts for a single categorical variable:
    
    .. plot::
        :context: close-figs
    
        >>> import seaborn as sns
        >>> sns.set(style="darkgrid")
        >>> titanic = sns.load_dataset("titanic")
        >>> ax = sns.countplot(x="class", data=titanic)
    
    Show value counts for two categorical variables:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.countplot(x="class", hue="who", data=titanic)
    
    Plot the bars horizontally:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.countplot(y="class", hue="who", data=titanic)
    
    Use a different color palette:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.countplot(x="who", data=titanic, palette="Set3")
    
    Use ``plt.bar`` keyword arguments for a different look:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.countplot(x="who", data=titanic,
        ...                    facecolor=(0, 0, 0, 0),
        ...                    linewidth=5,
        ...                    edgecolor=sns.color_palette("dark", 3))
    
    Use :func:`catplot` to combine a :func:`countplot` and a
    :class:`FacetGrid`. This allows grouping within additional categorical
    variables. Using :func:`catplot` is safer than using :class:`FacetGrid`
    directly, as it ensures synchronization of variable order across facets:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.catplot(x="class", hue="who", col="survived",
        ...                 data=titanic, kind="count",
        ...                 height=4, aspect=.7);
    [0;31mFile:[0m      ~/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py
    [0;31mType:[0m      function




```python
sns.countplot(
    x='sex', 
    data=titanic
)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff9a543eba8>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_25_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f207ef0>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_26_1.png)



```python
# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
```


```python
?sns.heatmap
```


    [0;31mSignature:[0m
    [0msns[0m[0;34m.[0m[0mheatmap[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mdata[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mvmin[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mvmax[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcmap[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcenter[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrobust[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mannot[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mfmt[0m[0;34m=[0m[0;34m'.2g'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mannot_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlinewidths[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlinecolor[0m[0;34m=[0m[0;34m'white'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcbar[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcbar_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcbar_ax[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msquare[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mxticklabels[0m[0;34m=[0m[0;34m'auto'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0myticklabels[0m[0;34m=[0m[0;34m'auto'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmask[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0max[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Plot rectangular data as a color-encoded matrix.
    
    This is an Axes-level function and will draw the heatmap into the
    currently-active Axes if none is provided to the ``ax`` argument.  Part of
    this Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``.
    
    Parameters
    ----------
    data : rectangular dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    center : float, optional
        The value at which to center the colormap when plotting divergant data.
        Using this parameter will change the default ``cmap`` if none is
        specified.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with robust quantiles instead of the extreme values.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as ``data``, then use this to annotate the heatmap instead
        of the raw data.
    fmt : string, optional
        String formatting code to use when adding annotations.
    annot_kws : dict of key, value mappings, optional
        Keyword arguments for ``ax.text`` when ``annot`` is True.
    linewidths : float, optional
        Width of the lines that will divide each cell.
    linecolor : color, optional
        Color of the lines that will divide each cell.
    cbar : boolean, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for `fig.colorbar`.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
    square : boolean, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    xticklabels, yticklabels : "auto", bool, list-like, or int, optional
        If True, plot the column names of the dataframe. If False, don't plot
        the column names. If list-like, plot these alternate labels as the
        xticklabels. If an integer, use the column names but plot only every
        n label. If "auto", try to densely plot non-overlapping labels.
    mask : boolean array or DataFrame, optional
        If passed, data will not be shown in cells where ``mask`` is True.
        Cells with missing values are automatically masked.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to ``ax.pcolormesh``.
    
    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.
    
    See also
    --------
    clustermap : Plot a matrix using hierachical clustering to arrange the
                 rows and columns.
    
    Examples
    --------
    
    Plot a heatmap for a numpy array:
    
    .. plot::
        :context: close-figs
    
        >>> import numpy as np; np.random.seed(0)
        >>> import seaborn as sns; sns.set()
        >>> uniform_data = np.random.rand(10, 12)
        >>> ax = sns.heatmap(uniform_data)
    
    Change the limits of the colormap:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.heatmap(uniform_data, vmin=0, vmax=1)
    
    Plot a heatmap for data centered on 0 with a diverging colormap:
    
    .. plot::
        :context: close-figs
    
        >>> normal_data = np.random.randn(10, 12)
        >>> ax = sns.heatmap(normal_data, center=0)
    
    Plot a dataframe with meaningful row and column labels:
    
    .. plot::
        :context: close-figs
    
        >>> flights = sns.load_dataset("flights")
        >>> flights = flights.pivot("month", "year", "passengers")
        >>> ax = sns.heatmap(flights)
    
    Annotate each cell with the numeric value using integer formatting:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.heatmap(flights, annot=True, fmt="d")
    
    Add lines between each cell:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.heatmap(flights, linewidths=.5)
    
    Use a different colormap:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.heatmap(flights, cmap="YlGnBu")
    
    Center the colormap at a specific value:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.heatmap(flights, center=flights.loc["January", 1955])
    
    Plot every other column label and don't plot row labels:
    
    .. plot::
        :context: close-figs
    
        >>> data = np.random.randn(50, 20)
        >>> ax = sns.heatmap(data, xticklabels=2, yticklabels=False)
    
    Don't draw a colorbar:
    
    .. plot::
        :context: close-figs
    
        >>> ax = sns.heatmap(flights, cbar=False)
    
    Use different axes for the colorbar:
    
    .. plot::
        :context: close-figs
    
        >>> grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        >>> f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        >>> ax = sns.heatmap(flights, ax=ax,
        ...                  cbar_ax=cbar_ax,
        ...                  cbar_kws={"orientation": "horizontal"})
    
    Use a mask to plot only part of a matrix
    
    .. plot::
        :context: close-figs
    
        >>> corr = np.corrcoef(np.random.randn(10, 200))
        >>> mask = np.zeros_like(corr)
        >>> mask[np.triu_indices_from(mask)] = True
        >>> with sns.axes_style("white"):
        ...     ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
    [0;31mFile:[0m      ~/anaconda3/lib/python3.7/site-packages/seaborn/matrix.py
    [0;31mType:[0m      function




```python
tita_cor = titanic.corr()
sns.heatmap(tita_cor)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff9a4171198>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_29_1.png)



```python

```




    <matplotlib.text.Text at 0x11d72da58>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_30_1.png)



```python
# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
```


```python
?sns.FacetGrid
```


    [0;31mInit signature:[0m
    [0msns[0m[0;34m.[0m[0mFacetGrid[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mdata[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrow[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcol[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcol_wrap[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msharex[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msharey[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mheight[0m[0;34m=[0m[0;36m3[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0maspect[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpalette[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrow_order[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcol_order[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue_order[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mhue_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdropna[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlegend_out[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdespine[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmargin_titles[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mxlim[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mylim[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msubplot_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgridspec_kws[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msize[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      Multi-plot grid for plotting conditional relationships.
    [0;31mInit docstring:[0m
    Initialize the matplotlib figure and FacetGrid object.
    
    This class maps a dataset onto multiple axes arrayed in a grid of rows
    and columns that correspond to *levels* of variables in the dataset.
    The plots it produces are often called "lattice", "trellis", or
    "small-multiple" graphics.
    
    It can also represent levels of a third varaible with the ``hue``
    parameter, which plots different subets of data in different colors.
    This uses color to resolve elements on a third dimension, but only
    draws subsets on top of each other and will not tailor the ``hue``
    parameter for the specific visualization the way that axes-level
    functions that accept ``hue`` will.
    
    When using seaborn functions that infer semantic mappings from a
    dataset, care must be taken to synchronize those mappings across
    facets. In most cases, it will be better to use a figure-level function
    (e.g. :func:`relplot` or :func:`catplot`) than to use
    :class:`FacetGrid` directly.
    
    The basic workflow is to initialize the :class:`FacetGrid` object with
    the dataset and the variables that are used to structure the grid. Then
    one or more plotting functions can be applied to each subset by calling
    :meth:`FacetGrid.map` or :meth:`FacetGrid.map_dataframe`. Finally, the
    plot can be tweaked with other methods to do things like change the
    axis labels, use different ticks, or add a legend. See the detailed
    code examples below for more information.
    
    See the :ref:`tutorial <grid_tutorial>` for more information.
    
    Parameters
    ----------
    data : DataFrame
        Tidy ("long-form") dataframe where each column is a variable and each
        row is an observation.    
    row, col, hue : strings
        Variables that define subsets of the data, which will be drawn on
        separate facets in the grid. See the ``*_order`` parameters to
        control the order of levels of this variable.
    col_wrap : int, optional
        "Wrap" the column variable at this width, so that the column facets
        span multiple rows. Incompatible with a ``row`` facet.    
    share{x,y} : bool, 'col', or 'row' optional
        If true, the facets will share y axes across columns and/or x axes
        across rows.    
    height : scalar, optional
        Height (in inches) of each facet. See also: ``aspect``.    
    aspect : scalar, optional
        Aspect ratio of each facet, so that ``aspect * height`` gives the width
        of each facet in inches.    
    palette : palette name, list, or dict, optional
        Colors to use for the different levels of the ``hue`` variable. Should
        be something that can be interpreted by :func:`color_palette`, or a
        dictionary mapping hue levels to matplotlib colors.    
    {row,col,hue}_order : lists, optional
        Order for the levels of the faceting variables. By default, this
        will be the order that the levels appear in ``data`` or, if the
        variables are pandas categoricals, the category order.
    hue_kws : dictionary of param -> list of values mapping
        Other keyword arguments to insert into the plotting call to let
        other plot attributes vary across levels of the hue variable (e.g.
        the markers in a scatterplot).
    legend_out : bool, optional
        If ``True``, the figure size will be extended, and the legend will be
        drawn outside the plot on the center right.    
    despine : boolean, optional
        Remove the top and right spines from the plots.
    margin_titles : bool, optional
        If ``True``, the titles for the row variable are drawn to the right of
        the last column. This option is experimental and may not work in all
        cases.    
    {x, y}lim: tuples, optional
        Limits for each of the axes on each facet (only relevant when
        share{x, y} is True.
    subplot_kws : dict, optional
        Dictionary of keyword arguments passed to matplotlib subplot(s)
        methods.
    gridspec_kws : dict, optional
        Dictionary of keyword arguments passed to matplotlib's ``gridspec``
        module (via ``plt.subplots``). Requires matplotlib >= 1.4 and is
        ignored if ``col_wrap`` is not ``None``.
    
    See Also
    --------
    PairGrid : Subplot grid for plotting pairwise relationships.
    relplot : Combine a relational plot and a :class:`FacetGrid`.
    catplot : Combine a categorical plot and a :class:`FacetGrid`.
    lmplot : Combine a regression plot and a :class:`FacetGrid`.
    
    Examples
    --------
    
    Initialize a 2x2 grid of facets using the tips dataset:
    
    .. plot::
        :context: close-figs
    
        >>> import seaborn as sns; sns.set(style="ticks", color_codes=True)
        >>> tips = sns.load_dataset("tips")
        >>> g = sns.FacetGrid(tips, col="time", row="smoker")
    
    Draw a univariate plot on each facet:
    
    .. plot::
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> g = sns.FacetGrid(tips, col="time",  row="smoker")
        >>> g = g.map(plt.hist, "total_bill")
    
    (Note that it's not necessary to re-catch the returned variable; it's
    the same object, but doing so in the examples makes dealing with the
    doctests somewhat less annoying).
    
    Pass additional keyword arguments to the mapped function:
    
    .. plot::
        :context: close-figs
    
        >>> import numpy as np
        >>> bins = np.arange(0, 65, 5)
        >>> g = sns.FacetGrid(tips, col="time",  row="smoker")
        >>> g = g.map(plt.hist, "total_bill", bins=bins, color="r")
    
    Plot a bivariate function on each facet:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="time",  row="smoker")
        >>> g = g.map(plt.scatter, "total_bill", "tip", edgecolor="w")
    
    Assign one of the variables to the color of the plot elements:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="time",  hue="smoker")
        >>> g = (g.map(plt.scatter, "total_bill", "tip", edgecolor="w")
        ...       .add_legend())
    
    Change the height and aspect ratio of each facet:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
        >>> g = g.map(plt.hist, "total_bill", bins=bins)
    
    Specify the order for plot elements:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])
        >>> g = g.map(plt.hist, "total_bill", bins=bins, color="m")
    
    Use a different color palette:
    
    .. plot::
        :context: close-figs
    
        >>> kws = dict(s=50, linewidth=.5, edgecolor="w")
        >>> g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",
        ...                   hue_order=["Dinner", "Lunch"])
        >>> g = (g.map(plt.scatter, "total_bill", "tip", **kws)
        ...      .add_legend())
    
    Use a dictionary mapping hue levels to colors:
    
    .. plot::
        :context: close-figs
    
        >>> pal = dict(Lunch="seagreen", Dinner="gray")
        >>> g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
        ...                   hue_order=["Dinner", "Lunch"])
        >>> g = (g.map(plt.scatter, "total_bill", "tip", **kws)
        ...      .add_legend())
    
    Additionally use a different marker for the hue levels:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
        ...                   hue_order=["Dinner", "Lunch"],
        ...                   hue_kws=dict(marker=["^", "v"]))
        >>> g = (g.map(plt.scatter, "total_bill", "tip", **kws)
        ...      .add_legend())
    
    "Wrap" a column variable with many levels into the rows:
    
    .. plot::
        :context: close-figs
    
        >>> att = sns.load_dataset("attention")
        >>> g = sns.FacetGrid(att, col="subject", col_wrap=5, height=1.5)
        >>> g = g.map(plt.plot, "solutions", "score", marker=".")
    
    Define a custom bivariate function to map onto the grid:
    
    .. plot::
        :context: close-figs
    
        >>> from scipy import stats
        >>> def qqplot(x, y, **kwargs):
        ...     _, xr = stats.probplot(x, fit=False)
        ...     _, yr = stats.probplot(y, fit=False)
        ...     plt.scatter(xr, yr, **kwargs)
        >>> g = sns.FacetGrid(tips, col="smoker", hue="sex")
        >>> g = (g.map(qqplot, "total_bill", "tip", **kws)
        ...       .add_legend())
    
    Define a custom function that uses a ``DataFrame`` object and accepts
    column names as positional variables:
    
    .. plot::
        :context: close-figs
    
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     data=np.random.randn(90, 4),
        ...     columns=pd.Series(list("ABCD"), name="walk"),
        ...     index=pd.date_range("2015-01-01", "2015-03-31",
        ...                         name="date"))
        >>> df = df.cumsum(axis=0).stack().reset_index(name="val")
        >>> def dateplot(x, y, **kwargs):
        ...     ax = plt.gca()
        ...     data = kwargs.pop("data")
        ...     data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
        >>> g = sns.FacetGrid(df, col="walk", col_wrap=2, height=3.5)
        >>> g = g.map_dataframe(dateplot, "date", "val")
    
    Use different axes labels after plotting:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="smoker", row="sex")
        >>> g = (g.map(plt.scatter, "total_bill", "tip", color="g", **kws)
        ...       .set_axis_labels("Total bill (US Dollars)", "Tip"))
    
    Set other attributes that are shared across the facetes:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="smoker", row="sex")
        >>> g = (g.map(plt.scatter, "total_bill", "tip", color="r", **kws)
        ...       .set(xlim=(0, 60), ylim=(0, 12),
        ...            xticks=[10, 30, 50], yticks=[2, 6, 10]))
    
    Use a different template for the facet titles:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="size", col_wrap=3)
        >>> g = (g.map(plt.hist, "tip", bins=np.arange(0, 13), color="c")
        ...       .set_titles("{col_name} diners"))
    
    Tighten the facets:
    
    .. plot::
        :context: close-figs
    
        >>> g = sns.FacetGrid(tips, col="smoker", row="sex",
        ...                   margin_titles=True)
        >>> g = (g.map(plt.scatter, "total_bill", "tip", color="m", **kws)
        ...       .set(xlim=(0, 60), ylim=(0, 12),
        ...            xticks=[10, 30, 50], yticks=[2, 6, 10])
        ...       .fig.subplots_adjust(wspace=.05, hspace=.05))
    [0;31mFile:[0m           ~/anaconda3/lib/python3.7/site-packages/seaborn/axisgrid.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m     




```python
ft = sns.FacetGrid(
    data=titanic,
    col='sex'
)
ft.map(sns.distplot, 'age', kde=None)
```




    <seaborn.axisgrid.FacetGrid at 0x7ff9864f1550>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_33_1.png)



```python

```




    <seaborn.axisgrid.FacetGrid at 0x11d81c240>




![png](07-Seaborn%20Exercises_files/07-Seaborn%20Exercises_34_1.png)


# Great Job!

### That is it for now! We'll see a lot more of seaborn practice problems in the machine learning section!
