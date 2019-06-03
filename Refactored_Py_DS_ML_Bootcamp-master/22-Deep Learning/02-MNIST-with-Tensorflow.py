#!/usr/bin/env python
# coding: utf-8

# # MNIST Data Set - Basic Approach

# ### Get the MNIST Data

# In[28]:


import tensorflow as tf


# In[29]:


from tensorflow.examples.tutorials.mnist import input_data


# In[30]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# ** Alternative sources of the data just in case: **
# 
# * http://yann.lecun.com/exdb/mnist/
# * https://github.com/mrgloom/MNIST-dataset-in-different-formats

# In[31]:


type(mnist)


# In[32]:


mnist.train.images


# In[33]:


mnist.train.num_examples


# In[34]:


mnist.test.num_examples


# In[35]:


mnist.validation.num_examples


# ### Visualizing the Data

# In[36]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


mnist.train.images[1].shape


# In[38]:


plt.imshow(mnist.train.images[1].reshape(28,28))


# In[39]:


plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')


# In[40]:


mnist.train.images[1].max()


# In[41]:


plt.imshow(mnist.train.images[1].reshape(784,1))


# In[42]:


plt.imshow(mnist.train.images[1].reshape(784,1),cmap='gist_gray',aspect=0.02)


# ## Create the Model

# In[43]:


x = tf.placeholder(tf.float32,shape=[None,784])


# In[44]:


# 10 because 0-9 possible numbers
W = tf.Variable(tf.zeros([784,10]))


# In[45]:


b = tf.Variable(tf.zeros([10]))


# In[46]:


# Create the Graph
y = tf.matmul(x,W) + b 


# Loss and Optimizer

# In[47]:


y_true = tf.placeholder(tf.float32,[None,10])


# In[48]:


# Cross Entropy


# In[49]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))


# In[50]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)


# In[51]:


train = optimizer.minimize(cross_entropy)


# ### Create Session

# In[52]:


init = tf.global_variables_initializer()


# In[53]:


with tf.Session() as sess:
    sess.run(init)
    
    # Train the model for 1000 steps on the training set
    # Using built in batch feeder from mnist for convenience
    
    for step in range(1000):
        
        batch_x , batch_y = mnist.train.next_batch(100)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        
    # Test the Train Model
    matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))


# While this may seem pretty good, we can actually do much better, the best models can get above 99% accuracy.
# 
