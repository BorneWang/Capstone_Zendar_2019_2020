#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:45:29 2019

@author: bowenwang
"""

import h5py
import numpy as np
from PIL import Image
from utils import calculate_norm
import os
import cv2 as cv

f1 = h5py.File('ffbp_output_good.h5','r')
#f2 = h5py.File('radar_images_new_quat_pos.h5','r')

aperture1 = f1['radar']['broad01']['aperture2D']
#aperture2 = f2['radar']['broad01']['aperture2D']
#img2_max = aperture2.attrs['max_value']
#img2_min = aperture2.attrs['min_value']
keys = list(aperture1.keys())
os.makedirs('linear', exist_ok=True)
os.makedirs('gamma', exist_ok=True)
os.makedirs('combine', exist_ok=True)
os.makedirs('combine2', exist_ok=True)
os.makedirs('dlinear', exist_ok=True)

#base = 20;
#i = 3
for i in range(10):
    image1 = aperture1[keys[i]]
    img_norm = calculate_norm(image1)
    img = img_norm
    img_max = np.nanmax(img)
    img_min = np.nanmin(img)
    img_255 = np.uint8(255*(img-img_min)/(img_max-img_min))
    
    '''
    # Method 1 : Linear
    img_linear = 4.0 * img_255
    img_linear[img_linear>255] = 255
    img_linear = np.around(img_linear)
    img_linear = img_linear.astype(np.uint8)
    img_linear = Image.fromarray(img_linear,'L')
    path = 'linear/' + str(i) + '_linear' + '.png'
    img_linear.save(path)
    '''
    
    '''
    # Method 2 : Gamma
    fi = img_255 / 255.0
    gamma = 0.57
    out = np.power(fi, gamma)
    img_gamma = np.zeros(out.shape, np.uint8)
    cv.normalize(out, img_gamma, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    img_gamma = Image.fromarray(img_gamma,'L')
    path = 'gamma/' + str(i) + '_gamma' + '.png'
    img_gamma.save(path)
    '''
    
    
    # Method 3 : Combine 1
    fi = img_255 / 255.0
    gamma = 0.57
    out = np.power(fi, gamma)
    img_gamma = np.zeros(out.shape, np.uint8)
    cv.normalize(out, img_gamma, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    img_linear = 2.0 * img_gamma # Change this number (4.0, 3.0, 2.0)
    img_linear[img_linear>255] = 255
    img_linear = np.around(img_linear)
    img_linear = img_linear.astype(np.uint8)
    img_linear = Image.fromarray(img_linear,'L')
    path = 'combine/' + str(i) + '_combine' + '.png'
    img_linear.save(path)
    
    '''
    # Method 4 : Combine 2
    img_linear = 4.0 * img_255
    img_linear[img_linear>255] = 255
    img_linear = np.around(img_linear)
    img_linear = img_linear.astype(np.uint8)
    fi = img_linear / 255.0
    gamma = 0.57
    out = np.power(fi, gamma)
    img_gamma = np.zeros(out.shape, np.uint8)
    cv.normalize(out, img_gamma, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    img_gamma = Image.fromarray(img_gamma,'L')
    path = 'combine2/' + str(i) + '_combine2' + '.png'
    img_gamma.save(path)
    '''
    
    '''
    # Method 5 : 
    threshold = 10
    h, w = img_255.shape[:2]
    out = np.zeros(img_255.shape, np.uint8)
    for j in range(h):
        for k in range(w):
            pix = img_255[j][k]
            if pix > threshold:
                out[j][k] = 40.0 * pix
            else:
                out[j][k] = pix
     
    out[out>255] = 255           
    out = np.around(out)
    out = out.astype(np.uint8)
    img_linear = Image.fromarray(out,'L')
    path = 'dlinear/' + str(i) + '_dlinear' + '.png'
    img_linear.save(path)
    '''
    
    
    
f1.close()