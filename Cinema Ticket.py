#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.metrics as metrics


# In[2]:


df = pd.read_csv(r"C:\Users\sbhsb\OneDrive\Desktop\cinemaTicket_Ref.csv")


# In[3]:


# Let's see the first 5 rows of the dataset
df.head()


# In[4]:


# Let's see the last 5 rows of the dataset
df.tail()


# In[5]:


df = df.drop('date', axis=1)


# In[6]:


df


# In[7]:


# Let's see the shape of the dataset
df.shape


# In[8]:


# Let's see the columns of the dataset
df.columns


# In[9]:


# Let's see the data types of the columns
df.dtypes


# In[10]:


# let's check the missing values
df.isnull().sum()


# In[11]:


# Let's drop the null values
df.dropna(inplace=True)


# In[12]:


# Let's check for missing values again

print("The null values are dropped")
df.isnull().sum()


# In[13]:


# Let's see the correlation between the columns
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True)


# In[14]:


#Assigning X values based on correlation with y
X = df[['ticket_price', 'occu_perc', 'show_time', 'tickets_sold','ticket_use','capacity']]
Y = df['total_sales']


# In[15]:


#Splitting the data into training and testing
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,random_state=42)


# In[16]:


lr = LinearRegression()
lr.fit(X_train,Y_train)


# In[17]:


#PREDICTING THE DATA
y_pred = lr.predict(X_test)


# In[18]:


plt.figure(figsize=(12,6))
plt.scatter(Y_test,y_pred,color='b')
plt.show()


# In[19]:


#Checking r2_score
r_squared = r2_score(Y_test, y_pred)
r_squared


# In[20]:


#Let's check for other metrics too
print('MAE: {}'.format(metrics.mean_absolute_error(Y_test, y_pred)))
print('MSE: {}'.format(metrics.mean_squared_error(Y_test, y_pred)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(Y_test, y_pred))))


# In[ ]:




