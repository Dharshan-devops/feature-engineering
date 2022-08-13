#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sklearn
import pandas as pd
import numpy as np


# In[22]:


data=pd.read_csv("titanic_train.csv")
data.head(5)


# In[23]:


null_count=data.isnull().sum()
null_count


# In[24]:


null_features=null_count[null_count>0].sort_values(ascending=False)
print(null_features)


# In[25]:


null_features=[features for features in data.columns if data[features].isnull().sum()>0]
print(null_features)


# In[26]:


data.shape


# In[30]:


print("{} unique values in pasenger id columns".format(len(data.passenger_id.unique())))


# In[32]:


print("{} unique values in Name columns".format(len(data.name.unique())))


# In[34]:


print("{} unique values in Ticket columns".format(len(data.ticket.unique())))


# In[60]:


data=pd.DataFrame(data)
data.head()


# In[62]:


data.shape


# In[ ]:


nn=data['age'].mean(axis=0,skipna=True)
nn=np.round(nn)
data['age']=data['age'].fillna(value=nn)
data=data.drop(['cabin'],axis=1)


# In[78]:


data.head()


# In[81]:


n1=data['embarked'].mode()
data['embarked']=data['embarked'].fillna(value='S')


# In[83]:


data.isnull().sum()


# In[86]:


data.head()


# In[90]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data['embarked']=label_encoder.fit_transform(data['embarked'])
data['sex']=label_encoder.fit_transform(data['sex'])


# In[92]:


data.head()


# In[ ]:


data=data.drop(['boat','body','home.dest'],axis=1)


# In[109]:


data.head()


# In[113]:


from sklearn.preprocessing import MinMaxScaler


# In[115]:


scaler=MinMaxScaler()
data[['age','fare']]=scaler.fit_transform(data[['age','fare']].values)


# In[117]:


data.head()


# In[121]:


from sklearn.preprocessing import MaxAbsScaler
maxabs=MaxAbsScaler()
data[['age','fare']]=maxabs.fit_transform(data[['age','fare']].values)
data.head()


# In[122]:


from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()
data[['age','fare']]=pt.fit_transform(data[['age','fare']].values)
data.head()

