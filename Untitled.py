#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
#import matplotlib.pyplot as plt

DATADIR = "C:\\train"
CATEGORIES = ["label2"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to cats or dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        Img_Width = 250
        Img_Height = 250


        new_array = cv2.resize(img_array, (Img_Width,Img_Height))
        cv2.imwrite(img,new_array)
        #plt.imshow(new_array,cmap = 'gray')
        #plt.show()


# In[ ]:




