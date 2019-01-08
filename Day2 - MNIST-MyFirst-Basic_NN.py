#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Getting the Data

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)


# In[3]:


type(mnist)


# In[4]:


mnist.train.images


# In[5]:


mnist.validation.images


# In[6]:


mnist.train.num_examples


# In[7]:


mnist.test.num_examples


# In[8]:


mnist.validation.num_examples


# ## Step 2: Visualizing the Data

# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


mnist.train.images.shape


# In[11]:


mnist.train.images[567]


# In[12]:


my_img = mnist.train.images[567].reshape(28, 28)


# In[13]:


plt.imshow(my_img)


# In[14]:


plt.imshow(my_img, cmap="gist_gray")


# In[15]:


my_img.min()


# In[16]:


my_img.max()


# In[17]:


my_img2 = mnist.train.images[567].reshape(784,1)
plt.imshow(my_img2)


# ## Step 3: Creating out NN Model
# 
# 1. Placeholders - 
# 2. Variables - 
# 3. Create out Computation Graph Operation - 
# 4. Loss Function - 
# 5. Optimixer - 
# 6. Create and run the session

# In[18]:


x = tf.placeholder(tf.float32, shape=[None, 784])


# In[19]:


# Weights, Bias 
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]


# # Create the graph 
# y = tf.matmul(x, W) + b

# In[22]:


y_true = tf.placeholder(tf.float32, shape = [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y))
# In[25]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(cross_entropy)
# In[27]:


init = tf.global_variables_initializer()


# In[ ]:


# Create and run Session 

with tf.Session() as sess:
    sess.run(init)
    
    # Train for say 10000 steps
    for step in range(10000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict = {x:batch_x, y_true:batch_y})
        
    #evaluate test train model 
    matches = tf.equal(tf.arg_max(y,1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
    print(sess.run(accuracy,feed_dict={x:mnist.text.images, y_true:mnist.text.labels}))
        

