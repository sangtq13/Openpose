#!/usr/bin/env python
# coding: utf-8

# # Suy đoán tư thế

# In[1]:


from PIL import Image
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch

test_image = './data/dancer.jpg'


# In[2]:


from utils.openpose_net import OpenPoseNet

net = OpenPoseNet()

# load weights of model
net_weights = torch.load(
    './weights/openpose_net_1.pth', map_location={'cuda:0': 'cpu'})
keys = list(net_weights.keys())

weights_load = {}


for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]
                 ] = net_weights[list(keys)[i]]

state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state)

print('load done')


# In[40]:


# Read image

oriImg = cv2.imread(test_image)  # B,G,R

# BGR->RGB
oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
plt.imshow(oriImg)
plt.show()

# Resize
size = (368, 368)
img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)
img = img.astype(np.float32) / 255.

# chuẩn hóa
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]

preprocessed_img = img.copy()  

for i in range(3):
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

# （height 、width、colors）→（colors、height、width）
img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

# cho thông tin vào tensor
img = torch.from_numpy(img)

x = img.unsqueeze(0)


# In[41]:


# Tạo heatmap
net.eval()
predicted_outputs, _ = net(x)

pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

pafs = cv2.resize(
    pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(
    heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
print(heatmaps)


# In[42]:


# Xem 1 số heatmap
heat_map = heatmaps[:, :, 6]  # 6: khửu tay
heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
heat_map = np.asarray(heat_map.convert('RGB'))

blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)
plt.imshow(blend_img)
plt.show()


heat_map = heatmaps[:, :, 7]  # 7はcổ tay
heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
heat_map = np.asarray(heat_map.convert('RGB'))

blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)
plt.imshow(blend_img)
plt.show()

# xem paf vectors
paf = pafs[:, :, 24]
paf = Image.fromarray(np.uint8(cm.jet(paf)*255))
paf = np.asarray(paf.convert('RGB'))


blend_img = cv2.addWeighted(oriImg, 0.5, paf, 0.5, 0)
plt.imshow(blend_img)
plt.show()


# In[43]:


from utils.decode_pose import decode_pose
_, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)


# In[44]:


# 結果を描画
plt.imshow(oriImg)
plt.show()

plt.imshow(result_img)
plt.show()


# In[ ]:




