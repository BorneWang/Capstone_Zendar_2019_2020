#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:01:48 2019

@author: bowenwang
"""
import numpy as np
import h5py
import os


def calculate_mirror_rotate(image):
    return np.rot90(np.fliplr(image),3)

def main():
    
    logger = open('gen_d5_mirror_rotate.logger.txt', 'w')
    
    f = h5py.File('radar_images_norm_log.h5','r')
    print("begin to processing *********************")
    print("begin to processing *********************", file = logger)
    aperture = f['radar']['broad01']['aperture2D']
    keys = list(aperture.keys())
    if os.path.exists('radar_images_mirror_rot.h5'):
        f_new = h5py.File('radar_images_mirror_rot.h5','r+')
        aperture_new = f_new['radar']['broad01']['aperture2D']
    else:
        f_new = h5py.File('radar_images_mirror_rot.h5','w')
        aperture_new = f_new.create_group("radar").create_group("broad01").create_group("aperture2D")
    max_value = -1
    min_value = 255
    f_new.close()
    
    '''
    !!!!!!!!!!!!!!!!!! change "start" | "np.linspace(50,4600,92)"!!!!!!!!!!!!!!!!!!
    '''
    start = 0
    for i in list(np.linspace(50,4600,92).astype('int')):
        images = list(map(lambda x:calculate_mirror_rotate(aperture[x]), keys[start:i]))
        f_new = h5py.File('radar_images_mirror_rot.h5','r+')
        aperture_new = f_new['radar']['broad01']['aperture2D']
        for j in range(start,i):
            idx = j - start
            image_new = aperture_new.create_dataset(keys[j],data=images[idx])
            max_value = max(max_value, np.nanmax(images[idx]))
            if np.nanmin(images[idx]) == float('-inf'):
                print("there is a -inf in file: ", keys[j] ,"index:",j, file = logger)
            min_value = min(min_value, max(np.nanmin(images[idx]),0))
            print("current image:",keys[j],"index:",j)
            print("current image:",keys[j],"index:",j, file = logger)   
            for attrs in list(aperture[keys[j]].attrs.keys()):
                if attrs == 'ATTITUDE':
                    image_new.attrs.create(attrs,np.array([(aperture[keys[j]].attrs[attrs][0][1],
                                                  aperture[keys[j]].attrs[attrs][0][2],
                                                  aperture[keys[j]].attrs[attrs][0][3],
                                                  aperture[keys[j]].attrs[attrs][0][0])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('w', '<f8')]))
                else:
                    image_new.attrs.create(attrs,aperture[keys[j]].attrs[attrs])
        start = i
        f_new.close()
        
    images = list(map(lambda x:calculate_mirror_rotate(aperture[x]), keys[start:]))
    f_new = h5py.File('radar_images_mirror_rot.h5','r+')
    aperture_new = f_new['radar']['broad01']['aperture2D']
    for i in range(start,len(keys)):
        idx = i - start
        image_new = aperture_new.create_dataset(keys[i],data=images[idx])
        max_value = max(max_value, np.nanmax(images[idx]))
        min_value = min(min_value, np.nanmin(images[idx]))
        print("current image:",keys[i],"index:",i)
        print("current image:",keys[i],"index:",i, file = logger)  
        for attrs in list(aperture[keys[i]].attrs.keys()):
            if attrs == 'ATTITUDE':
                image_new.attrs.create(attrs,np.array([(aperture[keys[i]].attrs[attrs][0][1],
                                                        aperture[keys[i]].attrs[attrs][0][2],
                                                        aperture[keys[i]].attrs[attrs][0][3],
                                                        aperture[keys[i]].attrs[attrs][0][0])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('w', '<f8')]))
            else:
                image_new.attrs.create(attrs,aperture[keys[i]].attrs[attrs])
                
    aperture_new.attrs.create('max_value',max_value)
    aperture_new.attrs.create('min_value',min_value)
    f_new.close()
    f.close()
    logger.close()
    
    

if __name__ == '__main__':
    main()
