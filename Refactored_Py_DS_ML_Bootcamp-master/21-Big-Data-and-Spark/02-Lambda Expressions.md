
#lambda expressions

One of Pythons most useful (and for beginners, confusing) tools is the lambda expression. lambda expressions allow us to create "anonymous" functions. This basically means we can quickly make ad-hoc functions without needing to properly define a function using def.

Function objects returned by running lambda expressions work exactly the same as those created and assigned by defs. There is key difference that makes lambda useful in specialized roles:

**lambda's body is a single expression, not a block of statements.**

* The lambda's body is similar to what we would put in a def body's return statement. We simply type the result as an expression instead of explicitly returning it. Because it is limited to an expression, a lambda is less general that a def. We can only squeeze design, to limit program nesting. lambda is designed for coding simple functions, and def handles the larger tasks.

Lets slowly break down a lambda expression by deconstructing a function:


```python
def square(num):
    result = num**2
    return result
```


```python
square(2)
```




    4



Continuing the breakdown:


```python
def square(num):
    return num**2
```


```python
square(2)
```




    4



We can actually write this in one line (although it would be bad style to do so)


```python
def square(num): return num**2
```


```python
square(2)
```




    4



This is the form a function that a lambda expression intends to replicate. A lambda expression can then be written as:


```python
lambda num: num**2
```




    <function __main__.<lambda>>



Note how we get a function back. We can assign this function to a label:


```python
square = lambda num: num**2
```


```python
square(2)
```




    4



And there you have it! The breakdown of a function into a lambda expression!
Lets see a few more examples:

##Example 1
Check it a number is even


```python
even = lambda x: x%2==0
```


```python
even(3)
```




    False




```python
even(4)
```




    True



##Example 2
Grab first character of a string:


```python
first = lambda s: s[0]
```


```python
first('hello')
```




    'h'



##Example 3
Reverse a string:


```python
rev = lambda s: s[::-1]
```


```python
rev('hello')
```




    'olleh'



##Example 4
Just like a normal function, we can accept more than one function into a lambda expresssion:


```python
adder = lambda x,y : x+y
```


```python
adder(2,3)
```




    5



lambda expressions really shine when used in conjunction with map(),filter() and reduce(). Each of those functions has its own lecture, so feel free to explore them if your very itnerested in lambda.

I highly recommend reading this blog post at [Python Conquers the Universe](https://pythonconquerstheuniverse.wordpress.com/2011/08/29/lambda_tutorial/) for a great breakdown on lambda expressions and some explanations of common confusions! 
