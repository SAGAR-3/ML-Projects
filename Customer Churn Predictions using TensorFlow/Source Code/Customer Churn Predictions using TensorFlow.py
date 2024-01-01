#!/usr/bin/env python
# coding: utf-8

# #STEP-1 IMPORTING DATA 

# In[75]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[76]:


df = pd.read_csv('Churn.csv')


# In[77]:


x = pd.get_dummies(df.drop(['Churn','Customer ID'],axis=1))


# In[78]:


for col in x.columns:
    print(col)


# In[79]:


x.head()


# In[80]:


y = df['Churn'].apply(lambda x:1 if x=='Yes' else 0)


# In[81]:


y.head()


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[83]:


X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int32)

X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.int32)


# In[84]:


X_train.head()


# In[85]:


y_train.head()


# #STEP-2 IMPORT DEPENDENCIES

# In[86]:


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


# #STEP-3 BUILD AND COMPILE MODEL 

# In[87]:


model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


# In[88]:


model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')


# #STEP-4 FIT PREDICT AND EVELUATE

# In[89]:


model.fit(X_train_tensor, y_train_tensor, epochs=200, batch_size=32) 


# In[98]:


y_hat = model.predict(X_test_tensor)
y_hat = [0 if val <0.5 else 1 for val in y_hat]


# In[91]:


y_hat


# In[99]:


accuracy_score(y_test_tensor, y_hat)


# #STEP-5 SAVING AND RELOADING THE MODEL 

# In[93]:


model.save('tfmodel')


# In[94]:


del model


# In[97]:


model = load_model('tfmodel')


# In[ ]:




