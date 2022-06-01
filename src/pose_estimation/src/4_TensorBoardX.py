#!/usr/bin/env python
# coding: utf-8

# # Hiển thị model bằng TensorBoardX
# 

# # Chuẩn bị
# 
# Cài thêm 2 thư viện tensorfliow và tensorboardx
# 
# 
# pip install tensorflow 
# 
# pip install tensorboardx
# 
# 
# 

# In[1]:


# import thư viện
import torch


# In[2]:


from utils.openpose_net import OpenPoseNet
# chuẩn bị model
net = OpenPoseNet()
net.train()


# In[3]:


# 1. gọi class để lưu giữtensorboardX
from tensorboardX import SummaryWriter
writer = SummaryWriter("./tbX/")


# 2. tạo dummy data để cho vào model
batch_size = 1
dummy_img = torch.rand(batch_size, 3, 368, 368)

# 3. lưu giữ liệu
writer.add_graph(net, (dummy_img, ))
writer.close()


# 4. mở commmand hoặc terminal chuyển đến folder pose_estimation và thực hiện câu lệnh

# tensorboard --logdir="./tbX/"

# sau đó access vào http://localhost:6006


# In[ ]:




