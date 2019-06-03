
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Matplotlib Exercises 

Welcome to the exercises for reviewing matplotlib! Take your time with these, Matplotlib can be tricky to understand at first. These are relatively simple plots, but they can be hard if this is your first time with matplotlib, feel free to reference the solutions as you go along.

Also don't worry if you find the matplotlib syntax frustrating, we actually won't be using it that often throughout the course, we will switch to using seaborn and pandas built-in visualization capabilities. But, those are built-off of matplotlib, which is why it is still important to get exposure to it!

** * NOTE: ALL THE COMMANDS FOR PLOTTING A FIGURE SHOULD ALL GO IN THE SAME CELL. SEPARATING THEM OUT INTO MULTIPLE CELLS MAY CAUSE NOTHING TO SHOW UP. * **

# Exercises

Follow the instructions to recreate the plots using this data:

## Data


```python
import numpy as np
x = np.arange(0,100)
y = x*2
z = x**2
```

** Import matplotlib.pyplot as plt and set %matplotlib inline if you are using the jupyter notebook. What command do you use if you aren't using the jupyter notebook?**


```python

```

## Exercise 1

** Follow along with these steps: **
* ** Create a figure object called fig using plt.figure() **
* ** Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax. **
* ** Plot (x,y) on that axes and set the labels and titles to match the plot below:**


```python

```




    <matplotlib.text.Text at 0x111534c50>




![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_5_1.png)


## Exercise 2
** Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively.**


```python

```


![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_7_0.png)


** Now plot (x,y) on both axes. And call your figure object to show it.**


```python

```




![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_9_0.png)



## Exercise 3

** Create the plot below by adding two axes to a figure object at [0,0,1,1] and [0.2,0.5,.4,.4]**


```python

```


![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_11_0.png)


** Now use x,y, and z arrays to recreate the plot below. Notice the xlimits and y limits on the inserted plot:**


```python

```




![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_13_0.png)




![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_13_1.png)


## Exercise 4

** Use plt.subplots(nrows=1, ncols=2) to create the plot below.**


```python

```


![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_15_0.png)


** Now plot (x,y) and (x,z) on the axes. Play around with the linewidth and style**


```python

```




![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_17_0.png)



** See if you can resize the plot by adding the figsize() argument in plt.subplots() are copying and pasting your previous code.**


```python

```




    <matplotlib.text.Text at 0x1141b4ba8>




![png](02-Matplotlib%20Exercises_files/02-Matplotlib%20Exercises_19_1.png)


# Great Job!
