#!/usr/bin/env python
# coding: utf-8

# In[25]:


import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def f(x):
    return x * 1
        
data_x, data_y = [], []
#i=0

for i in range(100):
#    i=i+1
    rnd = random.randrange(0, 100) 
    data_x.append([rnd])
    data_y.append([f(rnd)])
plt.plot(data_x, data_y, 'o')
plt.show()

#i=0
test_x, test_y = [], []
for i in range(100):
#    i=i+1
    rnd = random.randrange(0, 100)
    test_x.append([rnd])
    test_y.append([f(rnd)])
plt.show()


# In[8]:


test_x, test_y = [], []
for i in range(100):
    rnd = -10 + 20 * random.random()
    test_x.append([rnd])
    test_y.append([f(rnd)])
    
test_x = np.array(test_x)
test_y = np.array(test_y)

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, 1])
    Y = tf.placeholder(tf.float32, [None, 1])
    
    dim = 2048
    
    with tf.variable_scope('MLP'):
        net = tf.layers.dense(X, dim)
        net = tf.layers.dense(net, dim)
        net = tf.layers.dense(net, dim)
        net = tf.layers.dense(net, dim)
        net = tf.layers.dense(net, dim)     
        out = tf.layers.dense(net, 1)
        
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.square(Y - out))
        
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, l = sess.run([train, loss], feed_dict = {X: data_x, Y: data_y})
        print(i, l)
        
        if (i+1) % 100 == 0:
            saver.save(sess, 'logs/model.ckpt', global_step = i)
    saver.save(sess, 'logs/model.ckpt', global_step = i)


# In[ ]:





# In[12]:


test_result = []
with tf.Session(graph=g) as sess:
    checkpoint = tf.train.latest_checkpoint('logs')
    saver.restore(sess, checkpoint)
    for i in range(len(test_x)):
        result = sess.run(out, feed_dict = {X:[test_x[i]]})
        test_result.append(result[0][0])


# In[26]:


plt.plot(data_x, data_y, 'ro')
plt.plot(test_x, test_result, 'bo')
plt.show()


# In[27]:


with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, 1])
    Y = tf.placeholder(tf.float32, [None, 1])
    
    dim = 2048
    
    with tf.variable_scope('MLP'):
        net = tf.layers.dense(X, dim)
        net = tf.layers.dense(net, dim, activation=tf.nn.relu)
        net = tf.layers.dense(net, dim, activation=tf.nn.relu)
        net = tf.layers.dense(net, dim, activation=tf.nn.relu)
        net = tf.layers.dense(net, dim, activation=tf.nn.relu)
     
        out = tf.layers.dense(net, 1)
        
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.square(Y - out))
        
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()


# In[28]:


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, l = sess.run([train, loss], feed_dict = {X: data_x, Y: data_y})
        print(i, l)
        
        if (i+1) % 100 == 0:
            saver.save(sess, 'logs/model.ckpt', global_step = i)


# In[29]:


test_result = []
with tf.Session(graph=g) as sess:
    checkpoint = tf.train.latest_checkpoint('logs')
    saver.restore(sess, checkpoint)
    for i in range(len(test_x)):
        result = sess.run(out, feed_dict = {X:[test_x[i]]})
        test_result.append(result[0][0])


# In[30]:


plt.plot(data_x, data_y, 'ro')
plt.plot(test_x, test_result, 'bo')
plt.show()


# In[ ]:




