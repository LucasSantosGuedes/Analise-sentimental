#!/usr/bin/env python
# coding: utf-8

# In[3]:


##importando pandas e numpy
import pandas as pd
import numpy as np


# In[5]:


train = [
     'Eu te amo',
     'Você é algo assim... é tudo pra mim. Ao meu amor...Amor!',
     'Eu te odeio muito, você não presta!',
     'Não gosto de você'
]


# In[6]:


felling = [1,1,0,0]


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# In[8]:


vect.fit(train)


# In[9]:


##Examinando o dicionário criado em ordem alfabética
vect.get_feature_names()


# In[10]:


simple_train_dtm = vect.transform(train)
simple_train_dtm.toarray()


# In[14]:


df = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names(), index=train)
df


# In[15]:


type(simple_train_dtm)


# In[16]:


print(simple_train_dtm)


# In[17]:


novo_texto = ['te odeio']


# In[20]:


simple_test_dtm = vect.transform(novo_texto)

##Criando a visualização da matriz de ocorrência
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names(), index=novo_texto)


# In[21]:


##Importando o Classificador
from sklearn.neighbors import KNeighborsClassifier


# In[22]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(simple_train_dtm, felling)


# In[23]:


fell = knn.predict(simple_test_dtm)[0]


# In[24]:


fell


# In[26]:


if fell==1:
    print("Bom sentimento")
else:
    print("mal sentimento")


# In[28]:


sms = pd.read_table('sms.tsv', header=None, names=['label', 'message'])
sms.head(10)


# In[ ]:




