#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 


# In[3]:


df = pd.read_excel("dataset_2.xlsx")
df.head()


# In[4]:


y = df[df.columns[-1]]
y.head()


# In[6]:


x = df.drop(columns=["Plan"])


# In[7]:


x = x[x.columns[1:]]
x.head()


# In[8]:


escalador = StandardScaler()
x= escalador.fit_transform(x)
x


# In[9]:


x.shape[0]


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 2)


# In[11]:


x_train.shape


# In[12]:


n_entradas = x.shape[1]


# In[13]:


t_x_train = torch.from_numpy(x_train).float().to("cpu")
t_x_test = torch.from_numpy(x_test).float().to("cpu")
t_y_train = torch.from_numpy(y_train.values).float().to("cpu")
t_y_test = torch.from_numpy(y_test.values).float().to("cpu")
t_y_train = t_y_train[:,None] #
t_y_test = t_y_test[:,None]#que son estas linas y el .values


# In[14]:


t_y_train


# In[15]:


class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8, 1)
    
    def forward(self, inputs):
        pred_1 = torch.sigmoid(input=self.linear1(inputs))
        pred_2 = torch.sigmoid(input=self.linear2(pred_1))
        pred_f = torch.sigmoid(input=self.linear3(pred_2))
        return pred_f
    


# In[16]:


t_y_train[0]


# In[17]:


lr = 0.001
epochs = 2000
estatus_print = 100

model = Red(n_entradas=n_entradas)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

print("Entranando el modelo")
for epoch in range(1, epochs+1):
    y_pred= model(t_x_train)
    loss = loss_fn(input=y_pred, target=t_y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % estatus_print == 0:
        print(f"\nEpoch {epoch} \t Loss: {round(loss.item(), 4)}")
    
    with torch.no_grad():
        y_pred = model(t_x_test)
        y_pred_class = y_pred.round()
        correct = (y_pred_class == t_y_test).sum()
        accuracy = 100 * correct / float(len(t_y_test))
        if epoch % estatus_print == 0:
            print("Accuracy: {}".format(accuracy.item()))
    


# In[ ]:




