#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:01:48 2019

@author: bowenwang
"""
import numpy as np
import h5py
import os

def calculate_norm_log(image):
    row, col = image.shape
    new_image = np.zeros((row,col))
    for i in range(row):
        new_image[i,:] = list(map(lambda x: np.linalg.norm((x['real'],x['imag'])), 
                                  image[i,:]))
    log_image = np.log(new_image)
            
    return log_image

def main():
    
    logger = open('gen_d5_norm_log_logger.txt', 'a')
    
    f = h5py.File('ffbp_output_good.h5','r')
    print("begin to processing *********************")
    print("begin to processing *********************", file = logger)
    aperture = f['radar']['broad01']['aperture2D']
    keys = list(aperture.keys())
    if os.path.exists('radar_images_norm_log.h5'):
        f_new = h5py.File('radar_images_norm_log.h5','r+')
        aperture_new = f_new['radar']['broad01']['aperture2D']
    else:
        f_new = h5py.File('radar_images_norm_log.h5','w')
        aperture_new = f_new.create_group("radar").create_group("broad01").create_group("aperture2D")
        max_value, min_value = -1, float('Inf')
        aperture_new.attrs.create('max_value',max_value)
        aperture_new.attrs.create('min_value',min_value)
    max_value = aperture_new.attrs['max_value']
    min_value = aperture_new.attrs['min_value']
    f_new.close()
    '''
    
    count = 0
    for time in keys:
        count += 1
        print("current image:",time,"index:",count)
        image = calculate_norm_log(aperture[time])
        max_value = max(max_value, np.nanmax(image))
        min_value = min(min_value, np.nanmin(image))
        image_new = aperture_new.create_dataset(time,data=image)
        for attrs in list(aperture[time].attrs.keys()):
            image_new.attrs.create(attrs,aperture[time].attrs[attrs])
    '''
    
    
    '''
    !!!!!!!!!!!!!!!!!! change "start" | "np.linspace(4050,4600,12)"!!!!!!!!!!!!!!!!!!
    '''
    start = 4000
    for i in list(np.linspace(4050,4600,12).astype('int')):
        images = list(map(lambda x:calculate_norm_log(aperture[x]), keys[start:i]))
        f_new = h5py.File('radar_images_norm_log.h5','r+')
        aperture_new = f_new['radar']['broad01']['aperture2D']
        for j in range(start,i):
            idx = j - start
            image_new = aperture_new.create_dataset(keys[j],data=images[idx])
            print("current image:",keys[j],"index:",j)
            print("current image:",keys[j],"index:",j, file = logger)
            max_value = max(max_value, np.nanmax(images[idx]))
            min_value = min(min_value, np.nanmin(images[idx]))   
            for attrs in list(aperture[keys[j]].attrs.keys()):
                image_new.attrs.create(attrs,aperture[keys[j]].attrs[attrs])
        start = i
        aperture_new.attrs['max_value'] = max_value
        aperture_new.attrs['min_value'] = min_value
        f_new.close()
        
    images = list(map(lambda x:calculate_norm_log(aperture[x]), keys[start:]))
    f_new = h5py.File('radar_images_norm_log.h5','r+')
    aperture_new = f_new['radar']['broad01']['aperture2D']
    for i in range(start,len(keys)):
        idx = i - start
        image_new = aperture_new.create_dataset(keys[i],data=images[idx])
        print("current image:",keys[i],"index:",i)
        print("current image:",keys[i],"index:",i, file = logger)
        max_value = max(max_value, np.nanmax(images[idx]))
        min_value = min(min_value, np.nanmin(images[idx]))   
        for attrs in list(aperture[keys[i]].attrs.keys()):
            image_new.attrs.create(attrs,aperture[keys[i]].attrs[attrs])
            
            
    aperture_new.attrs['max_value'] = max_value
    aperture_new.attrs['min_value'] = min_value
    f_new.close()
    f.close()
    logger.close()
    
    

if __name__ == '__main__':
    main()
