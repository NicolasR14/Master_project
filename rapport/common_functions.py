import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# from ROI_extraction import preprocess_image
import cv2
import os
def HSV_features_generation(dataset_path):
    ids = []
    labels = {}
    classes = {'excess':1,'normal':0,'insufficient':2}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path) :
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_path, filename) 
                    ids.append(img_path)
                    labels[img_path]=classes[class_name]

    params = {'n_classes': 3}

    y = np.empty((len(ids)), dtype=int)
    X = []
    for i, ID in enumerate(ids):
        img= cv2.imread(ID)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H_val,S_val,V_val = img[:,:,0],img[:,:,1],img[:,:,2]
        # Extract the mean values of hue, saturation, and value
        H_mean = np.mean(H_val)
        S_mean = np.mean(S_val)
        V_mean = np.mean(V_val)
        # Extract the standard deviation of hue, saturation, and value
        H_std = np.std(H_val)
        S_std = np.std(S_val)
        V_std = np.std(V_val)
        X.append([H_mean,H_std,S_mean,S_std,V_mean,V_std])
        y[i] = labels[ID]
    return pd.DataFrame(X,columns=['H_mean','H_std','S_mean','S_std','V_mean','V_std']),y