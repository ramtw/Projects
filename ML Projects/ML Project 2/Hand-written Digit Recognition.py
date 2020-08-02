#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784')


# In[2]:


mnist


# In[3]:


x , y = mnist['data'] , mnist['target']


# In[4]:


x[0]


# In[5]:


y


# In[6]:


x.shape


# In[7]:


y.shape


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib
import matplotlib.pyplot as plt


# In[10]:


some_digit = x[65000]
some_digit_image = some_digit.reshape(28,28)


# In[11]:


some_digit_image


# In[12]:


plt.imshow(some_digit_image , cmap = matplotlib.cm.binary , interpolation='nearest')
plt.axis('off')


# In[13]:


y[65000]


# In[14]:


x_train , x_test = x[:60000] , x[60000:]


# In[15]:


y_train , y_test = y[:60000] , y[60000:]


# In[16]:


import numpy as np
shuffle_index = np.random.permutation(60000)
x_train , y_train = x_train[shuffle_index] , y_train[shuffle_index]


# ## Creating a 2 detector

# In[17]:


y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_test_2 = (y_test == 2)
y_train_2 = (y_train == 2)


# In[18]:


y_train_2


# In[19]:


y_test_2


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


clf = LogisticRegression()


# In[22]:


clf.fit(x_train , y_train_2)


# In[23]:


clf.predict([some_digit])


# In[25]:


from sklearn.model_selection import cross_val_score
a = cross_val_score( clf , x_train , y_train_2 , cv=3 , scoring='accuracy')


# In[27]:


a.mean()


# In[ ]:




