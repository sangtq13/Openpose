#!/usr/bin/env python
# coding: utf-8

# ## Chuẩn bị data

# In[1]:


import os
import urllib.request
import zipfile
import tarfile


# In[2]:


data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)


# In[3]:



weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)


# In[ ]:


# Download MSCOCOの2014 Val images [41K/6GB]

url =  "http://images.cocodataset.org/zips/val2014.zip"
target_path = os.path.join(data_dir, "val2014.zip") 

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)
    
    zip = zipfile.ZipFile(target_path)
    zip.extractall(data_dir)  
    zip.close()
    


# In[6]:


# Download COCO.json and input into data folder


# https://www.dropbox.com/s/0sj2q24hipiiq5t/COCO.json?dl=0


# In[7]:


# Download mask data and input into data foloder

# https://www.dropbox.com/s/bd9ty7b4fqd5ebf/mask.tar.gz?dl=0


# In[8]:


# giải nén tar gz
save_path = os.path.join(data_dir, "mask.tar.gz") 

with tarfile.open(save_path, 'r:*') as tar:
    tar.extractall(data_dir)


# In[ ]:


# download model and input to weights

https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?dl=0
    


# 以上
