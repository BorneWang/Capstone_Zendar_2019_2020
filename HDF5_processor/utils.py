#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 20:43:08 2019

@author: bowenwang
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

#def generate_video(image_dir):
    

def generate_png(f,image_dir):
    aperture = f['radar']['broad01']['aperture2D']
    keys = list(aperture.keys())
    os.makedirs(image_dir, exist_ok=True)
    for time in keys:
        #generate 2D matrix
        image = calculate_norm_log(aperture[time])
        
        #generate heatmap
        fig, ax = plt.subplots()
        height, width = image.shape 
        plt.imshow(image)
        plt.axis('off')
        height, width = image.shape 
        fig.set_size_inches(width/100.0/4.0, height/100.0/4.0) 
        plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
        plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
        plt.margins(0,0)
        #save heatmap
        path = image_dir + '/' + time.replace('.','_') + '.png'
        plt.savefig(path, dpi=400)
        

def generate_npy(f,image_dir='', return_=False):
    aperture = f['radar']['broad01']['aperture2D']
    keys = list(aperture.keys())
    images = list(map(lambda x:calculate_norm_log(aperture[x]), keys))
    if return_ == True:
        return images, keys
    else:
        os.makedirs(image_dir, exist_ok=True)
        for i in range(len(keys)):
            path = image_dir + '/' + keys[i].replace('.','_')
            np.save(path,images[i])
    

def Load_src(src):
    return h5py.File(src,'r')


def calculate_norm_log(image):
    row, col = image.shape
    new_image = np.zeros((row,col))
    for i in range(row):
        new_image[i,:] = list(map(lambda x: np.linalg.norm((x['real'],x['imag'])), 
                                  image[i,:]))
    log_image = np.log(new_image)
            
    return log_image