#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd 
import numpy as np
from time import time 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# In[21]:


df = pd.read_excel('Dataset.xlsx')
df


# In[22]:


print(df.dtypes)


# In[23]:


df=df.drop('Nombre ',axis=1)
df


# In[24]:


y = df[['tipo de dieta']]
x =df.drop('tipo de dieta',axis=1)


# In[25]:


#from sklearn.preprocessing import MinMaxScaler


# In[26]:


#scale = MinMaxScaler(feature_range=(0,100))
#df['altura']=scale.fit_transform(df['altura'].values.reshape(-1,1))
#df.head()


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(
x,
y,
test_size =0.3)


# In[29]:


modelo = SVC(kernel ='linear')


# In[31]:


hora_inicio = time()
modelo.fit(x_train.values,y_train.values.ravel())
print('entrenamiento terminado en {}'.format(time()-hora_inicio))


# In[32]:


hora_inicio = time()
y_pred = modelo.predict(x_test.values)
print('prediccion terminada en {}'.format(time()-hora_inicio))


# In[33]:


prec = accuracy_score(y_test,y_pred)
print(f'prec: {prec}')


# In[37]:


x_prueba = pd.DataFrame({'edad':[30,44,15,60],'altura':[170,188,165,162],'peso':[70,105,52,60],'sexo':[1,1,1,0],'entrena':[1,0,1,0],})
x_prueba


# In[38]:


modelo.predict(x_prueba)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




