#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:51:28 2019

@author: bowenwang
"""

import numpy as np
import h5py
from scipy.spatial.transform import Rotation as Rot

def increase_contrast(heatmap,min_magnitude,max_magnitude):
    gamma = 0.57
    lin_coeff = 3 # Change this number (4.0, 3.0, 2.0)
        
    heatmap = (np.exp(heatmap)-np.exp(min_magnitude))/(np.exp(max_magnitude)-np.exp(min_magnitude))
    heatmap = np.power(heatmap, gamma)*255
    # thresholding between 10 and 11
    heatmap = thresholding(heatmap, lin_coeff, 16.3)
    heatmap = heatmap.astype(np.uint8)
    return heatmap
    
def thresholding(img, lincoe, thres):
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i,j] > thres:
                img[i,j] = lincoe*img[i,j] + 60
    return img


def main(): 
    
    f = h5py.File('radar_images_new_quat_pos.h5','r')
    aperture = f['radar']['broad01']['aperture2D']
    keys = list(aperture.keys())
    max_value = aperture.attrs['max_value']
    min_value = aperture.attrs['min_value']
    
    #creat new H5 file
    f_new = h5py.File('radar_images_increaseC_pos.h5','w')
    aperture_new = f_new.create_group("radar").create_group("broad01").create_group("aperture2D")
    max_value_new = 0
    min_value_new = 256
    
    count = 0
    
    for time in keys:
        print(count)
        count += 1
        # Create new dataset using time
        image = aperture[time]
        image_data = increase_contrast(image[...],min_value,max_value)
        max_value_new = max(max_value_new, np.nanmax(image_data))
        min_value_new = min(min_value_new, np.nanmin(image_data))
        image_new = aperture_new.create_dataset(time,data=image_data)
        image_new.attrs.create('TIMESTAMP_SPAN',image.attrs['TIMESTAMP_SPAN'])
        image_new.attrs.create('APERTURE_SPAN',image.attrs['APERTURE_SPAN'])
        image_new.attrs.create('ATTITUDE',image.attrs['ATTITUDE'])
        
        
        r2 = Rot.from_quat(list(image.attrs['ATTITUDE'][0])) #from Global to CV2!
        
        p_topright_global = list(image.attrs['POSITION'][0]) #top right (incorrect top left) in Global
        p_topleft_global = p_topright_global + r2.inv().apply([-30,0,0]) #topleft in Global, new POSITION!
        
        image_new.attrs.create('POSITION', np.array([(p_topleft_global[0],p_topleft_global[1],
                                                      p_topleft_global[2])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]))
        
    
    aperture_new.attrs['max_value'] = max_value_new
    aperture_new.attrs['min_value'] = min_value_new
    f.close()
    f_new.close()
    

if __name__ == '__main__':
    main()