#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:28:25 2019

@author: bowenwang
"""

import h5py
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
from statistics import stdev
from sklearn.cluster import DBSCAN

def DBSCAN_filter(im, kernel, scale, binary=True):
    """ Filter images to binary based on DBSCAN clustering """
    blur1 = cv2.GaussianBlur(im, kernel, scale)
    ret1,th1 = cv2.threshold(blur1,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    X = np.transpose(np.nonzero(th1))
    db = DBSCAN(eps=5.0, min_samples=30).fit(X)
    np.place(th1, th1, db.labels_ > -1)
    print("binary is :", binary)
    if binary:        
        return (255*th1).astype(np.uint8)
    else:
        return np.multiply(im, th1).astype(np.uint8)

class Preprocessor:
    def __init__(self, src, goal, groundtruth, log = True, loaddata = False, init_load = 0, DBSCAN = True, mean = -1, std = -1):
        # load files
        self.f = h5py.File(src,'r')
        self.aperture = self.f['radar']['squint_left_facing']['aperture2D']
        self.keys = list(self.aperture.keys())
        # create write-in data
        self.f_new = h5py.File(goal,'w')
        self.aperture_new = self.f_new.create_group("radar").create_group("broad01").create_group("aperture2D")
        
        # set temp images list and temp data
        self.images = []
        
        # options
        self.goal =goal
        self.log = log
        self.loaddata = loaddata
        self.init_load = init_load
        self.DBSCAN = DBSCAN
        self.gt = groundtruth
        self.mean = mean
        self.std = std
        
    def RUN(self):
        if self.loaddata:
            self.run_from_loaddata()
        else:
            self.run_from_scratch()
          
        if not self.log:
            print("just do magnitude, finished")
            return
        
        # do global normalization, DBSCAN filtering and copy attrs
        self.new_preprocessing()
        
        # add flag
        self.aperture_new.attrs.create('preprocessed', True)
        
        # copy tracklog
        tracklog1 = self.f['tracklog']
        self.f_new.create_dataset('tracklog',data=tracklog1[...])
        # get translation from tracklog to gps
        self.tracklog_trans(self.goal)
        self.f.close()
        self.f_new.close()
        
        # add groundtruth to the file
        self.adding_groundtruth()
            
    def run_from_loaddata(self):
        print("run from loaddata")
        self.load_backup_data()      
        
    def run_from_scratch(self):
        print("run from scratch")
        # do magnitude and save bachup files
        self.magnitude_and_save()      
    
    def magnitude_and_save(self):
        # process images in 50-size batch
        batch = 50
        ite = len(self.keys)//batch
        start = 0
        for i in list(np.linspace(batch,ite*batch,ite).astype('int')):
            # back-up file
            name = 'backup/'+str(start)+'.h5'
            f_backup = h5py.File(name,'w')
            aperture_backup = f_backup.create_group("radar").create_group("broad01").create_group("aperture2D")
            
            # calculate norm, do mirror and rotate 90 degree
            tic = time.time()
            images = list(map(lambda x:self.do_norm_mirror_rotate(self.aperture[x]), self.keys[start:i]))
            for j in range(start,i):
                idx = j - start
                if self.log:
                    self.images.append(np.log(images[idx]))
                else:
                    self.images.append(images[idx])
                
                # back-up file
                aperture_backup.create_dataset(self.keys[j],data=images[idx])
                
            start = i
            toc = time.time()
            print("batch number:",i,"/",len(self.keys),"time:",toc-tic)
            
            # close backup file
            f_backup.close()
            
        # residual images
        name = 'backup/'+'rest'+'.h5'
        f_backup = h5py.File(name,'w')
        aperture_backup = f_backup.create_group("radar").create_group("broad01").create_group("aperture2D")
        images = list(map(lambda x:self.do_norm_mirror_rotate(self.aperture[x]), self.keys[start:len(self.keys)]))
        for j in range(start,len(self.keys)):
            idx = j - start
            if self.log:
                self.images.append(np.log(images[idx]))
            else:
                self.images.append(images[idx])
            
            # back-up file
            aperture_backup.create_dataset(self.keys[j],data=images[idx])

        print("total images:",j+1,"Finished!")
        f_backup.close()
                
    def new_preprocessing(self):
        # check correct:
        if len(self.images) == len(self.keys):
            print("list length correct, continue")
        else:
            print("list length error, stop")
            return
        
        # get gloabl mean
        if self.mean > 0:
            global_mean = self.mean
            print("global mean is:", global_mean)
        else:
            img_count = 1
            sum_ = 0.0
            count = 0
            for img in self.images:
                row, col = img.shape
                for i in range(row):
                    for j in range(col):
                        if img[i][j] > 0:
                            sum_ += img[i][j]
                            count += 1
                print("img count:", img_count)
                img_count += 1
            global_mean = sum_/count
            print("global mean is:", global_mean)
        
        if self.std > 0:
            global_std = self.std
            print("global_std is:", global_std)
        else:
            # get std
            img_count = 1
            sum_ = 0.0
            for img in self.images:
                row, col = img.shape
                for i in range(row):
                    for j in range(col):
                        if img[i][j] > 0:
                            sum_ += (img[i][j]-global_mean)**2
                print("img count:", img_count)
                img_count += 1
            global_std = (sum_/count)**(0.5)
            print("global_std is:", global_std)
        
        for i in range(len(self.keys)):
            heatmap = self.images[i]
            heatmap = ((heatmap-global_mean)/global_std)*255.0/4.0
            heatmap[heatmap < 0] = 0
            heatmap[heatmap > 255] = 255
            heatmap = heatmap.astype(np.uint8)
            
            
            # DBSCAN filtering
            if self.DBSCAN:
                if np.nanmax(heatmap) > 0:
                    heatmap = DBSCAN_filter(heatmap, kernel=(9,9), scale=0, binary=False)
                    print("DBSCAN procedure:", i)
            
            #save
            self.image_new = self.aperture_new.create_dataset(self.keys[i],data=heatmap)
            # add attrs
            self.adding_attrs(i)
    
            
    def tracklog_trans(self, goal):
        tklog = Tracklog(goal)
        self.aperture_new.attrs.create('tracklog_translation',tklog.translations_POV_mean)
        
        
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
    
    def load_backup_data(self):
        batch = 50
        ite = len(self.keys)//batch
        for i in list(np.linspace(0,(ite-1)*batch,ite).astype('int')):
            path = 'backup_vn_0217\\' + str(i) + '.h5'
            print('back up file:', path)
            f_backup = h5py.File(path,'r')
            aperture_backup = f_backup['radar']['broad01']['aperture2D']
            keys_backup = list(aperture_backup.keys())
            for j in keys_backup:
                img = aperture_backup[j]
                if self.log:
                    self.images.append(np.log(img[...]))
                else:
                    self.images.append(img[...])

            f_backup.close()
        
        path = 'backup_vn_0217\\' + 'rest' + '.h5'
        print('back up file:', path)
        f_backup = h5py.File(path,'r')
        aperture_backup = f_backup['radar']['broad01']['aperture2D']
        keys_backup = list(aperture_backup.keys())

        for j in keys_backup:
            img = aperture_backup[j]
            if self.log:
                self.images.append(np.log(img[...]))
            else:
                self.images.append(img[...])

        print("total images:",len(self.images),"Finished!")
        f_backup.close()
        
        
    def adding_groundtruth(self):
        # read data
        f1 = h5py.File(self.goal,'r+')
        f2 = h5py.File(self.gt,'r')        

        aperture2 = f2['radar']['squint_left_facing']['aperture2D']

        broad1 = f1['radar']['broad01']
        groundtruth = broad1.create_group('groundtruth')

        keys = list(aperture2.keys())

        # copy and precessing position and attitude

        for key in keys:
            img = aperture2[key]
            img_new = groundtruth.create_dataset(key,data=0)
            
            
            r0 = Rot.from_quat(list((img.attrs['ATTITUDE'][0][1],
                                     img.attrs['ATTITUDE'][0][2],
                                     img.attrs['ATTITUDE'][0][3],
                                     img.attrs['ATTITUDE'][0][0]))) # From POV(up, left) to ECEF
            r0_inv = r0.inv() # from ECEF to POV(up,left)
            r1 = Rot.from_dcm([[0,-1,0],[-1,0,0],[0,0,-1]]) #from POV to CV2 Coordinate(right,down)
            r2 = r1*r0_inv #from ECEF to CV2, new ATTITUDE!
            new_quat = r2.as_quat() # new ATTITUDE to be saved!
            # position
            p_bottomright_global = list(img.attrs['POSITION'][0]) #bottom right in ECEF
            p_topleft_global = p_bottomright_global + r0.apply([20,30,0]) #topleft in ECEF, new POSITION!
            # Or p_topleft_global = p_bottomleft_global + r2.inv().apply([-30,-20,0])
            # save to h5
            img_new.attrs.create('ATTITUDE', np.array([(new_quat[0], new_quat[1],
                                                        new_quat[2], new_quat[3])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('w', '<f8')]))
            img_new.attrs.create('POSITION', np.array([(p_topleft_global[0],p_topleft_global[1],
                                                        p_topleft_global[2])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]))

        f1.close()
        f2.close()
        
        tklog = Tracklog(self.goal, foldername = 'groundtruth', tracklog = True, value = self.gt)
        f1 = h5py.File(self.goal,'r+')
        groundtruth = f1['radar']['broad01']['groundtruth']
        groundtruth.attrs.create('tracklog_translation',tklog.translations_POV_mean)
        f1.close()
    
    
class Tracklog:
    
    def __init__(self, src, foldername = 'aperture2D', tracklog = False, value = ''):
        self.translations_ECEF = dict()
        self.translations_POV = dict()
        self.translation_value = 0
        self.position_x = []
        self.position_y = []
        self.translations_POV_mean = 0
        self.translations_POV_stdev = 0
        self.foldername = foldername
        self.tracklog = tracklog
        self.value = value
        
        tic = time.time()
        self.loaddata(src)
        self.get_translations(src)
        print("time consume:", time.time()-tic)
        
    def loaddata(self, src):
        # load h5 file
        hdf5 = h5py.File(src,'r')
        # radar image data
        aperture = hdf5['radar']['broad01'][self.foldername]
        times = list(aperture.keys())
        N_img = len(times)
        print("radar images :", N_img)
        # tracklog data
        if self.tracklog:
            f_gt = h5py.File(self.value, 'r')
            tracklog = f_gt['tracklog']
        else:
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
        aperture = hdf5['radar']['broad01'][self.foldername]
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
        