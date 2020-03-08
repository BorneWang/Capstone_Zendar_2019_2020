#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:31:30 2020

@author: bowenwang
"""
import cv2
import numpy as np
#import h5py
from sklearn.cluster import DBSCAN
#from copy import deepcopy


#f = h5py.File('3VXs.h5','r')
#aperture = f['radar']['broad01']['aperture2D']
#keys = list(aperture.keys())

def DBSCAN_filter(im):
    row, col = im.shape
    blur1 = cv2.GaussianBlur(im,(9,9),1)
    ret1,th1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2) #adaptive
    
    
    X = np.zeros((1, 2))
    for i in range(row):
        for j in range(col):
            if th1[i][j] == 255:
                X = np.insert(X, 0, np.array([i,j]),axis=0)
            
    db = DBSCAN(eps=2.0, min_samples=10).fit(X)
    X = X.astype(np.int32)

    for i in range(len(X)):
        if db.labels_[i] == -1:
            th1[X[i][0]][X[i][1]] = 0
            blur1[X[i][0]][X[i][1]] = 0
            im[X[i][0]][X[i][1]] = 0
            #X = np.delete(X,i,0)
    
    return im


# without increasing constrast: do thresholding and no thresholding
# iteration 625 try everything
# try different parameters
# 0 to 400
# the difference between bad and good gps: should be const
# 10 to 75
# 10 to 40 cv2 good
    

