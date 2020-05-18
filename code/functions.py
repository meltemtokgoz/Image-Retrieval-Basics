import os
import cv2 
import numpy as np
from matplotlib import pyplot as plt

#This code get image file name*************************************************
query_path  = "dataset/query"
train_path  = "dataset/train"

query_img_names = []
train_img_names = []

for (dirpath, dirnames, filenames) in os.walk(query_path):
    for f in filenames:
        query_img_names.append(os.path.join(os.path.join(dirpath, f)))

for (dirpath, dirnames, filenames) in os.walk(train_path):
    for f in filenames:
        train_img_names.append(os.path.join(os.path.join(dirpath, f)))

#******************************************************************************
#This is SIFT functions 

def ToGray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray
    
def sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    mean_shıft_vector = np.mean(desc,axis=0)
    return kp, desc, mean_shıft_vector
    
def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

#******************************************************************************