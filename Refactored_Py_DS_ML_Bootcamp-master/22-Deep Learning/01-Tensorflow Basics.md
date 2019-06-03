
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# TensorFlow Basics

Remember to reference the video for full explanations, this is just a notebook for code reference.

You can import the library:


```python
import tensorflow as tf
```


```python
print(tf.__version__)
```

    1.10.0


### Simple Constants

Let's show how to create a simple constant with Tensorflow, which TF stores as a tensor object:


```python
hello = tf.constant('Hello World')
```


```python
type(hello)
```




    tensorflow.python.framework.ops.Tensor




```python
x = tf.constant(100)
```


```python
type(x)
```




    tensorflow.python.framework.ops.Tensor



### Running Sessions

Now you can create a TensorFlow Session, which is a class for running TensorFlow operations.

A `Session` object encapsulates the environment in which `Operation`
objects are executed, and `Tensor` objects are evaluated. For example:


```python
sess = tf.Session()
```


```python
sess.run(hello)
```




    b'Hello World'




```python
type(sess.run(hello))
```




    bytes




```python
sess.run(x)
```




    100




```python
type(sess.run(x))
```




    numpy.int32



## Operations

You can line up multiple Tensorflow operations in to be run during a session:


```python
x = tf.constant(2)
y = tf.constant(3)
```


```python
with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(x+y))
    print('Subtraction',sess.run(x-y))
    print('Multiplication',sess.run(x*y))
    print('Division',sess.run(x/y))
```

    Operations with Constants
    Addition 5
    Subtraction -1
    Multiplication 6
    Division 0.6666666666666666


### Placeholder

You may not always have the constants right away, and you may be waiting for a constant to appear after a cycle of operations. **tf.placeholder** is a tool for this. It inserts a placeholder for a tensor that will be always fed.

**Important**: This tensor will produce an error if evaluated. Its value must be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`. For example, for a placeholder of a matrix of floating point numbers:

    x = tf.placeholder(tf.float32, shape=(1024, 1024))

Here is an example for integer placeholders:


```python
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
```


```python
x
```




    <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=int32>




```python
type(x)
```




    tensorflow.python.framework.ops.Tensor



### Defining Operations


```python
add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)
```

Running operations with variable input:


```python
d = {x:20,y:30}
```


```python
with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(add,feed_dict=d))
    print('Subtraction',sess.run(sub,feed_dict=d))
    print('Multiplication',sess.run(mul,feed_dict=d))
```

    Operations with Constants
    Addition 50
    Subtraction -10
    Multiplication 600


Now let's see an example of a more complex operation, using Matrix Multiplication. First we need to create the matrices:


```python
import numpy as np
# Make sure to use floats here, int64 will cause an error.
a = np.array([[5.0,5.0]])
b = np.array([[2.0],[2.0]])
```


```python
a
```




    array([[5., 5.]])




```python
a.shape
```




    (1, 2)




```python
b
```




    array([[2.],
           [2.]])




```python
b.shape
```




    (2, 1)




```python
mat1 = tf.constant(a)
```


```python
mat2 = tf.constant(b)
```

The matrix multiplication operation:


```python
matrix_multi = tf.matmul(mat1,mat2)
```

Now run the session to perform the Operation:


```python
with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)
```

    [[20.]]


That is all for now! Next we will expand these basic concepts to construct out own Multi-Layer Perceptron model!

# Great Job!
