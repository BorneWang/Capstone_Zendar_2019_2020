#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:29:34 2019

@author: bowenwang
"""
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as Rot



def main():
    
    #logger = open('gen_d5_change_qua_pos.txt', 'w')
    
    f = h5py.File('radar_norm_log_mirror_rot.h5','r')
    #print("begin to processing *********************")
    #print("begin to processing *********************", file = logger)
    aperture = f['radar']['broad01']['aperture2D']
    keys = list(aperture.keys())
    
    #creat new H5 file
    f_new = h5py.File('radar_images_new_quat_pos.h5','w')
    aperture_new = f_new.create_group("radar").create_group("broad01").create_group("aperture2D")
    aperture_new.attrs.create('max_value',aperture.attrs['max_value'])
    aperture_new.attrs.create('min_value',aperture.attrs['min_value'])
    
    for time in keys:
        # Create new dataset using time
        image = aperture[time]
        image_new = aperture_new.create_dataset(time,data=image)
        image_new.attrs.create('TIMESTAMP_SPAN',image.attrs['TIMESTAMP_SPAN'])
        image_new.attrs.create('APERTURE_SPAN',image.attrs['APERTURE_SPAN'])
        
        
        r0 = Rot.from_quat(list(image.attrs['ATTITUDE'][0])) #from driver's POV to Global
        r0_inv = r0.inv() # r0's inv : from Global to driver's POV
        r1 = Rot.from_dcm([[0,-1,0],[-1,0,0],[0,0,-1]]) #from POV to CV2 Coordinate
        r2 = r1*r0_inv #from Global to CV2, new ATTITUDE!
        new_quat = r2.as_quat() # new ATTITUDE!
        
        p_bottomleft_global = list(image.attrs['POSITION'][0]) #bottom left in Global
        p_topleft_global = p_bottomleft_global + r0.apply([20,0,0]) #topleft in Global, new POSITION!
        
        image_new.attrs.create('ATTITUDE', np.array([(new_quat[0], new_quat[1],
                                                  new_quat[2], new_quat[3])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('w', '<f8')]))
        image_new.attrs.create('POSITION', np.array([(p_topleft_global[0],p_topleft_global[1],
                                                      p_topleft_global[2])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]))
        
    
    
    f.close()
    #logger.close()
    
    

if __name__ == '__main__':
    main()