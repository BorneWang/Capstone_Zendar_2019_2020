#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:28:25 2019

@author: bowenwang
"""

import h5py
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
from statistics import stdev

class hdf5_processor:
    def __init__(self, src, goal):
        # load files
        f = h5py.File(src,'r')
        self.aperture = f['radar']['broad01']['aperture2D']
        self.keys = list(self.aperture.keys())
        # create write-in data
        f_new = h5py.File(goal,'w')
        self.aperture_new = f_new.create_group("radar").create_group("broad01").create_group("aperture2D")
        
        # set temp images list and temp data
        self.images = []
        self.max = -1
        self.min = float('Inf')
        
        # do process
        self.processdata()
        self.increse_contrast()
        
        # copy tracklog
        tracklog1 = f['tracklog']
        tracklog2 = f_new.create_dataset('tracklog',data=tracklog1[...])
        
        f.close()
        f_new.close()
        
        # calculate tracklog translation
        self.tracklog_trans(goal)
    
    def processdata(self):
        # process images in 100-size batch
        ite = len(self.keys)//100
        start = 0
        for i in list(np.linspace(100,ite*100,ite).astype('int')):
            # calculate norm, do mirror and rotate 90 degree
            tic = time.time()
            images = list(map(lambda x:self.do_norm_mirror_rotate(self.aperture[x]), self.keys[start:i]))
            for j in range(start,i):
                idx = j - start
                self.images.append(images[idx])
                # save max and min
                self.max = max(self.max, np.nanmax(images[idx]))
                self.min = min(self.min, np.nanmin(images[idx])) 
            start = i
            toc = time.time()
            print("batch number:",i,"/",len(self.keys),"time:",toc-tic)
            
        # residual images
        images = list(map(lambda x:self.do_norm_mirror_rotate(self.aperture[x]), self.keys[start:len(self.keys)]))
        for j in range(start,len(self.keys)):
            idx = j - start
            self.images.append(images[idx])
            # save max and min
            self.max = max(self.max, np.nanmax(images[idx]))
            self.min = min(self.min, np.nanmin(images[idx]))
        print("total images:",j+1,"Finished!")
        
        # save max_value and min_value
        self.aperture_new.attrs.create('max_value',self.max)
        self.aperture_new.attrs.create('min_value',self.min)
                
    def increse_contrast(self):
        # check correct:
        if len(self.images) == len(self.keys):
            print("list lenght correct, continue")
        else:
            return
        # increase contrast
        gamma = 0.57
        lin_coeff = 3 # Change this number (4.0, 3.0, 2.0)
        for i in range(len(self.keys)):
            heatmap = self.images[i]
            heatmap = (heatmap - self.min)/(self.max - self.min)
            heatmap = np.power(heatmap, gamma)*255
            # thresholding between 10 and 11
            heatmap = self.thresholding(heatmap, lin_coeff, 16.3)
            heatmap = heatmap.astype(np.uint8)
            
            # save to h5 file
            self.image_new = self.aperture_new.create_dataset(self.keys[i],data=heatmap)
            
            # add attrs
            self.adding_attrs(i)
            
    def tracklog_trans(self, goal):
        tklog = Tracklog(goal)
        self.aperture_new.attrs.create('tracklog_translation',tklog.translations_POV_mean)
        
    def thresholding(self, img, lincoe, thres):
        row, col = img.shape
        for i in range(row):
            for j in range(col):
                if img[i,j] > thres:
                    img[i,j] = lincoe*img[i,j] + 60
        return img
        
    def adding_attrs(self, idx):
        old_image = self.aperture[self.keys[idx]]
        r0 = Rot.from_quat(list((old_image.attrs['ATTITUDE'][0][1],
                                     old_image.attrs['ATTITUDE'][0][2],
                                     old_image.attrs['ATTITUDE'][0][3],
                                     old_image.attrs['ATTITUDE'][0][0]))) # From POV(up, left) to ECEF
        r0_inv = r0.inv() # from ECEF to POV(up,left)
        r1 = Rot.from_dcm([[0,-1,0],[-1,0,0],[0,0,-1]]) #from POV to CV2 Coordinate(right,down)
        r2 = r1*r0_inv #from ECEF to CV2, new ATTITUDE!
        new_quat = r2.as_quat() # new ATTITUDE to be saved!
        # position
        p_bottomright_global = list(old_image.attrs['POSITION'][0]) #bottom right in ECEF
        p_topleft_global = p_bottomright_global + r0.apply([20,30,0]) #topleft in ECEF, new POSITION!
        # Or p_topleft_global = p_bottomleft_global + r2.inv().apply([-30,-20,0])
        # save to h5
        self.image_new.attrs.create('ATTITUDE', np.array([(new_quat[0], new_quat[1],
                                              new_quat[2], new_quat[3])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('w', '<f8')]))
        self.image_new.attrs.create('POSITION', np.array([(p_topleft_global[0],p_topleft_global[1],
                                                  p_topleft_global[2])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]))
    
    
        # copy other attrs
        self.image_new.attrs.create('TIMESTAMP_SPAN',old_image.attrs['TIMESTAMP_SPAN'])
        self.image_new.attrs.create('APERTURE_SPAN',old_image.attrs['APERTURE_SPAN'])
        
    def do_norm_mirror_rotate(self,image):
        return self.calculate_mirror_rotate(self.calculate_norm(image))
        
    def calculate_norm(self, image):
        row, col = image.shape
        new_image = np.zeros((row,col))
        for i in range(row):
            new_image[i,:] = list(map(lambda x: np.linalg.norm((x['real'],x['imag'])), 
                                  image[i,:]))
            
        return new_image
    
    def calculate_mirror_rotate(self, image):
        return np.rot90(np.fliplr(image),3)
    
    
class Tracklog:
    
    def __init__(self, src):
        self.translations_ECEF = dict()
        self.translations_POV = dict()
        self.translation_value = 0
        self.position_x = []
        self.position_y = []
        self.translations_POV_mean = 0
        self.translations_POV_stdev = 0
        
        tic = time.time()
        self.loaddata(src)
        self.get_translations(src)
        print("time consume:", time.time()-tic)
        
    def loaddata(self, src):
        # load h5 file
        hdf5 = h5py.File(src,'r')
        # radar image data
        aperture = hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys())
        N_img = len(times)
        print("radar images :", N_img)
        # tracklog data
        tracklog = hdf5['tracklog']
        timestamp = tracklog['timestamp']
        position_x = tracklog['position']['x']
        position_y = tracklog['position']['y']
        position_z = tracklog['position']['z']
        self.position_x = interp1d(timestamp, position_x)
        self.position_y = interp1d(timestamp, position_y)
        self.position_z = interp1d(timestamp, position_z)
        N_log = len(tracklog)
        print("tracklog units:", N_log)   
        hdf5.close()            
        
    def get_translations(self, src):
        #load file
        hdf5 = h5py.File(src,'r')
        # radar image data
        aperture = hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys())
        N_img = len(times)
        # define list
        trans_list = []
        POV_x_list = []
        POV_y_list = []
        POV_z_list = []
        # loop
        for key in range(N_img):
            try:
                car_pos_x = self.position_x(float(times[key]))
                car_pos_y = self.position_y(float(times[key]))
                car_pos_z = self.position_z(float(times[key]))
            except:
                break
            radar_pos = aperture[times[key]].attrs['POSITION'][0]
            radar_att = (aperture[times[key]].attrs['ATTITUDE'][0][0],
                         aperture[times[key]].attrs['ATTITUDE'][0][1],
                         aperture[times[key]].attrs['ATTITUDE'][0][2],
                         aperture[times[key]].attrs['ATTITUDE'][0][3])
            radar_att = Rot.from_quat(radar_att)
            self.translations_ECEF[key] = (radar_pos[0] - car_pos_x,
                                           radar_pos[1] - car_pos_y,
                                           radar_pos[2] - car_pos_z)
            self.translations_POV[key] = radar_att.apply(self.translations_ECEF[key])
            POV_x_list.append(self.translations_POV[key][0])
            POV_y_list.append(self.translations_POV[key][1])
            POV_z_list.append(self.translations_POV[key][2])
            trans_list.append(np.linalg.norm(self.translations_ECEF[key]))
        
        # get results
        self.translation_value = sum(trans_list)/len(trans_list)
        self.translations_POV_mean = (sum(POV_x_list)/len(POV_x_list), 
                                      sum(POV_y_list)/len(POV_y_list),
                                      sum(POV_z_list)/len(POV_z_list))
        self.translations_POV_stdev = (stdev(POV_x_list),
                                       stdev(POV_y_list),
                                       stdev(POV_z_list))
        
            
        hdf5.close()
        