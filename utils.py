from sklearn.metrics import confusion_matrix
from statistics import mean
from torch.nn import Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(config, mode):
   # data_dir = '/data/users1/reihaneh/projects/alzheimer_schizophrenia/TC/data/splits/train_r'
   # test_dir= '/data/users1/reihaneh/projects/alzheimer_schizophrenia/TC/data/splits/test_r'
    
    if mode == 'train':    
        #trn_path = os.path.join(data_dir, str(config.r)+'s'+str(config.s)+'.pkl')
        #data = pickle.load(open(f'/'+(trn_path), 'rb'))
        data = pickle.load(open(f'/data/users1/reihaneh/projects/alzheimer_schizophrenia/TC/data/splits/train_r0s0.pkl',"rb"))
        print("in utils_train")
    else:

        #tst_path = os.path.join(test_dir, str(config.r)+'s'+str(config.s)+'.pkl')
        #data = pickle.load(open(f'/'+(tst_path), 'rb'))
        data = pickle.load(open('/data/users1/reihaneh/projects/alzheimer_schizophrenia/TC/data/splits/test_r0s0.pkl',"rb"))
        print("in utils_test")

    return data

def preprocess(data):
    data = data[['SubID','Label']]
    ica_path = "/data/users1/reihaneh/projects/alzheimer_schizophrenia/TC/data/original/"
    data['ica']= " "
    for i in range(len(data)):
        data['ica'][i] = np.load(ica_path+data['SubID'][i]+".npy")

    data = data.drop(['SubID'], axis=1)
    
    x = data['ica']
    y = data['Label']
    
    for i in range(len(x)):
        x.loc[i] = x.loc[i].T
        
    abc = [69,53,98,99,45,21,56,3,9,2,11,27,54,66,80,72,16,5,62,15,12,93,20,8,77,68,33,43,70,61,55,63,79,84,96,88,48,81,37,67,38,83,32,40,23,71,17,51,94,13,18,4,7]
    d = []

    for i in range(1,101):
        if i not in abc:
            d.append(i-1)
    
    X = np.zeros((len(data), 157, 100))
    for i in range(len(x)):
        X[i] = x[i]

    X = torch.from_numpy(X)
    a = np.delete(X,d, axis=2)

    y = pd.get_dummies(y, drop_first = True)
    Y = torch.LongTensor(y.to_numpy())
    print("in utils_preprocess")
    return a, Y

def split_data(X,y):
    x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)
    print("in utils split")


    return x_train, x_val, y_train, y_val




