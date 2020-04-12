import os
import cv2
import h5py
import datetime
import numpy as np
from copy import deepcopy
from data import RadarData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj, increase_saturation, merge_img, data_projection, projection

class Map():
    
    def __init__(self, name = None):
        self.name = name
        self.img_cov = 10 # covariance of each the pixel value in an image
        
        self.chunk_size = 1000        
        self.display ={'text': "0 ; 0", 'img': None, 'overlay_fig':None, 'overlay': None, 'scale': 1, 'pos': np.array([0,0,0]), 'axes': None, 'fig':None, 'gps_pos':None} 
        
        if name is None:
            # Create a black map
            self.map_name = 'map_'+str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")
            print("Creating map: map_"+str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")+'.h5')
            os.makedirs(os.path.dirname('maps/'+self.map_name+'.h5'), exist_ok=True)
            hdf5 = h5py.File('maps/'+self.map_name+'.h5','a')
            try:
                map_hdf5 = hdf5.create_group("map")
                cov_map_hdf5 = hdf5.create_group("covariance")
                ini_map = np.nan*np.ones((self.chunk_size, self.chunk_size))
                ini_cov = np.nan*np.ones((self.chunk_size, self.chunk_size))
                map_hdf5.create_dataset("0/0", data = ini_map, shape=(self.chunk_size, self.chunk_size) )        
                cov_map_hdf5.create_dataset("0/0", data = ini_cov, shape=(self.chunk_size, self.chunk_size) ) 
                self.precision = None
                self.gps_pos = None
                self.attitude = None
            finally:
                hdf5.close()
        else:     
            # Retrieve an already existing map by name
            self.map_name = name
            hdf5 = h5py.File('maps/'+self.map_name+'.h5','r')
            try:
                map_hdf5 = hdf5["map"]
                self.precision = map_hdf5.attrs["PRECISION"]
                self.gps_pos = map_hdf5.attrs["POSITION"]
                self.attitude = rot.from_quat(map_hdf5.attrs["ATTITUDE"])
            finally:
                hdf5.close()

    def init_map(self, radardata):
        """ Initialize the map with an initial radardata """
        hdf5 = h5py.File('maps/'+self.map_name+'.h5','a')
        try:
            map_hdf5 = hdf5["map"]
            
            map_hdf5.attrs["POSITION"] = radardata.gps_pos
            map_hdf5.attrs["ATTITUDE"] = radardata.attitude.as_quat()
            map_hdf5.attrs["PRECISION"] = radardata.precision
        finally:
            hdf5.close()
        
        self.precision = radardata.precision    
        self.gps_pos = deepcopy(radardata.gps_pos) 
        self.attitude = deepcopy(radardata.attitude)

    def set_img_covariance(self, cov):
        """ Set the covariance of an image added to the map """
        self.img_cov = cov

    def build_partial_map(self, otherdata):
        """ Build partial map of the chunks that are needed to contain the new data """
        hdf5 = h5py.File('maps/'+self.map_name+'.h5','a')
        try:
            map_hdf5 = hdf5["map"]
            cov_map_hdf5 = hdf5["covariance"]
            
            q = rotation_proj(self.attitude, otherdata.attitude)
            P5 = self.attitude.apply(otherdata.gps_pos - self.gps_pos)[0:2]
            P6 = P5 + q.apply(np.array([otherdata.width(),0,0]))[0:2]
            P7 = P5 + q.apply(np.array([otherdata.width(),otherdata.height(),0]))[0:2]
            P8 = P5 + q.apply(np.array([0,otherdata.height(),0]))[0:2]
            
            P9 = np.array([min(P5[0],P6[0],P7[0],P8[0]),min(P5[1],P6[1],P7[1],P8[1])])
            P10 = np.array([max(P5[0],P6[0],P7[0],P8[0]),max(P5[1],P6[1],P7[1],P8[1])])
            
            chunk_1 = np.flip(np.floor((P9/self.precision)/self.chunk_size).astype(np.int))
            chunk_2 = np.flip(np.floor((P10/self.precision)/self.chunk_size).astype(np.int))
            
            img = np.nan*np.ones((self.chunk_size*(1+chunk_2[0]-chunk_1[0]), self.chunk_size*(1+chunk_2[1]-chunk_1[1])))
            cov_img = np.nan*np.ones((self.chunk_size*(1+chunk_2[0]-chunk_1[0]), self.chunk_size*(1+chunk_2[1]-chunk_1[1])))
            
            for i in range(chunk_1[0], chunk_2[0]+1):
                for j in range(chunk_1[1], chunk_2[1]+1):
                    if not str(i)+"/"+str(j) in map_hdf5:
                        map_hdf5.create_dataset(str(i)+"/"+str(j), data = np.nan*np.ones((self.chunk_size, self.chunk_size)), shape=(self.chunk_size, self.chunk_size) )        
                        cov_map_hdf5.create_dataset(str(i)+"/"+str(j), data = np.nan*np.ones((self.chunk_size, self.chunk_size)), shape=(self.chunk_size, self.chunk_size) ) 
                    img[(i-chunk_1[0])*self.chunk_size:(i-chunk_1[0]+1)*self.chunk_size, (j-chunk_1[1])*self.chunk_size:(j-chunk_1[1]+1)*self.chunk_size] = map_hdf5[str(i)+"/"+str(j)]
                    cov_img[(i-chunk_1[0])*self.chunk_size:(i-chunk_1[0]+1)*self.chunk_size, (j-chunk_1[1])*self.chunk_size:(j-chunk_1[1]+1)*self.chunk_size] = cov_map_hdf5[str(i)+"/"+str(j)]
            gps_pos = self.gps_pos + self.attitude.inv().apply(np.array([chunk_1[1]*self.chunk_size*self.precision, chunk_1[0]*self.chunk_size*self.precision, 0]))
        finally:            
            hdf5.close()
        return img, cov_img, gps_pos
    
    def update_map(self, img, cov_img, pos):
        """ Updating part of the map with a given image """
        hdf5 = h5py.File('maps/'+self.map_name+'.h5','a')
        map_hdf5 = hdf5["map"]
        cov_map_hdf5 = hdf5["covariance"]
        
        chunk = np.flip(np.round((self.attitude.apply(pos - self.gps_pos)[0:2]/self.precision)/self.chunk_size).astype(np.int))
        for i in range(chunk[0], chunk[0]+int(np.ceil(np.shape(img)[0]/self.chunk_size))):
            for j in range(chunk[1], chunk[1]+int(np.ceil(np.shape(img)[1]/self.chunk_size))):
                map_hdf5[str(i)+"/"+str(j)][...] = img[(i-chunk[0])*self.chunk_size:(i-chunk[0]+1)*self.chunk_size, (j-chunk[1])*self.chunk_size:(j-chunk[1]+1)*self.chunk_size]
                cov_map_hdf5[str(i)+"/"+str(j)][...] = cov_img[(i-chunk[0])*self.chunk_size:(i-chunk[0]+1)*self.chunk_size, (j-chunk[1])*self.chunk_size:(j-chunk[1]+1)*self.chunk_size]
        hdf5.close()
                
    def add_data(self, otherdata):
        """ Fusionning a new radardata with part of the map """
        if self.gps_pos is None:
            self.init_map(otherdata)
        
        img1, cov_img1, new_origin = self.build_partial_map(otherdata)
        shape = np.shape(img1)
        
        v2 = self.attitude.apply(otherdata.gps_pos - new_origin)[0:2]/self.precision
        M2 = np.concatenate((rotation_proj(self.attitude, otherdata.attitude).as_dcm()[:2,:2],np.array([[v2[0]],[v2[1]]])), axis = 1)
        img2 = cv2.warpAffine(otherdata.img, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        mask = cv2.warpAffine(np.ones(np.shape(otherdata.img)), M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0);
        diff = mask - np.ones(shape)
        diff[diff != 0] = np.nan
        img2 = diff + img2
        
        cov_img2 = cv2.warpAffine(self.img_cov*np.ones(np.shape(otherdata.img)), M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        cov_img2 = diff + cov_img2

        img, cov_img = merge_img(img1, img2, cov_img1, cov_img2)
        self.update_map(img, cov_img, new_origin)
        return img1, img2, v2
        
    def show(self, gps_pos = None, overlay=None):
        """ Show a matplotlib representation of the map 
            gps_pos: GPS pos where to show the map, origin of the map is used if not specified
        """
        # Parameter for the map display
        speed_trans = 100
        speed_scroll = 0.1
        shape = (1000, 2000)
        scroll_limit = 0.4
        overlay_alpha = 0.5
        border = 2
                
        def press(event):
            if event.key == 'left':
                self.display['pos'] = self.display['pos'] - np.array([self.display['scale']*speed_trans*self.precision,0,0])
            elif event.key == 'right':
                self.display['pos'] = self.display['pos'] + np.array([self.display['scale']*speed_trans*self.precision,0,0])
            elif event.key == 'up':
                self.display['pos'] = self.display['pos'] - np.array([0,self.display['scale']*speed_trans*self.precision,0])
            elif event.key == 'down':               
                self.display['pos'] = self.display['pos'] + np.array([0,self.display['scale']*speed_trans*self.precision,0])
            center = -self.precision*np.array([0.5*shape[1], 0.5*shape[0],0])
            img, _ = self.extract_from_map(self.display['gps_pos']+self.attitude.apply(self.display['pos']+center,True), self.attitude, shape, self.display['scale'])
            self.display['img'].set_data(increase_saturation(np.nan_to_num(img)))
            
            if not self.display['overlay'] is None:
                img_overlay = np.nan_to_num(self.display['overlay'].predict_image(self.display['gps_pos'] + self.attitude.apply(self.display['pos']+center,True), self.attitude, (int(np.ceil(shape[0]/self.display['scale'])), int(np.ceil(shape[1]/self.display['scale'])))))
                overlay_red = np.zeros((np.shape(img_overlay)[0], np.shape(img_overlay)[1], 4))
                overlay_red[:,:,0] = img_overlay
                overlay_red[:,:,3] = (img_overlay != 0)*overlay_alpha*255
                self.display['overlay_fig'].set_data(increase_saturation(overlay_red.astype(np.uint8)))           
            
            self.display['text'].set_text(str(round(self.display['pos'][0],2))+" ; "+ str(round(self.display['pos'][1],2)))
            plt.draw()
            
        def scroll(event):
            if event.step != 0:
                if self.display['scale'] - speed_scroll*event.step > scroll_limit:     
                    self.display['scale'] = self.display['scale'] - speed_scroll*event.step
                else:
                    self.display['scale'] = scroll_limit
                    
            center = -self.precision*np.array([0.5*shape[1], 0.5*shape[0],0])
            img, _ = self.extract_from_map(self.display['gps_pos']+self.attitude.apply(self.display['pos'] + center, True), self.attitude, shape, self.display['scale'])           
            self.display['img'].set_data(increase_saturation(np.nan_to_num(img)))
            
            if not self.display['overlay'] is None:
                img_overlay = np.nan_to_num(self.display['overlay'].predict_image(self.display['gps_pos']+self.attitude.apply(self.display['pos'] + center,True), self.attitude, (int(np.ceil(shape[0]/self.display['scale'])), int(np.ceil(shape[1]/self.display['scale'])))))
                overlay_red = np.zeros((np.shape(img_overlay)[0], np.shape(img_overlay)[1], 4))
                overlay_red[:,:,0] = img_overlay
                overlay_red[:,:,3] = (img_overlay != 0)*overlay_alpha*255
                self.display['overlay_fig'].set_data(increase_saturation(overlay_red.astype(np.uint8)))
            
            plt.draw()
            
        def close(event):
            self.display['fig'] = None
              
        missing = (self.display['fig'] is None)
        if gps_pos is None:
            self.display['gps_pos'] = deepcopy(self.gps_pos)
        else:
            if len(gps_pos)==2:
                self.display['gps_pos'] = deepcopy(self.gps_pos)
                self.display['pos'] = np.append(gps_pos, 0)
            else:   
                self.display['gps_pos'] = projection(self.gps_pos, self.attitude, gps_pos)
                if missing:
                    self.display['pos'] = np.array([0,0,0])
        
        center = -self.precision*np.array([0.5*shape[1], 0.5*shape[0],0])
        if not overlay is None:
            self.display['overlay'] = data_projection(self.gps_pos, self.attitude, overlay)
            img_border = 255*np.ones(np.shape(overlay.img))
            img_border[border:-border,border:-border] = overlay.img[border:-border,border:-border]
            self.display['overlay'].img = img_border
        else:
            self.display['overlay'] = None
        if missing:
            plt.close(self.map_name)
            if self.display['fig'] is None:
                self.display['text'] = "0 ; 0"
                self.display['scale'] = 1
                self.display['fig'] = plt.figure(num=self.map_name, facecolor=(1,1,1))
                self.display['fig'].canvas.mpl_connect('key_press_event', press)
                self.display['fig'].canvas.mpl_connect('scroll_event', scroll)
                self.display['fig'].canvas.mpl_connect('close_event', close)
                self.display['axes'] = plt.axes()
                self.display['axes'].set_facecolor("black")
                self.display['axes'].get_xaxis().set_visible(False)
                self.display['axes'].get_yaxis().set_visible(False)
            if self.gps_pos is None:
                img = np.nan*np.ones(shape)
            else:
                img, _ = self.extract_from_map(self.display['gps_pos'] + self.attitude.apply(self.display['pos'] + center,True), self.attitude, shape, self.display['scale'])
            self.display['img'] = self.display['axes'].imshow(increase_saturation(np.nan_to_num(img)), cmap='gray', vmin=0, vmax=255, zorder=1)
            if not self.display['overlay'] is None:
                img_overlay = np.nan_to_num(self.display['overlay'].predict_image(self.display['gps_pos'] + self.attitude.apply(self.display['pos']+ center,True), self.attitude, (int(np.ceil(shape[0]/self.display['scale'])), int(np.ceil(shape[1]/self.display['scale'])))))
                overlay_red = np.zeros((np.shape(img_overlay)[0], np.shape(img_overlay)[1], 4))
                overlay_red[:,:,0] = img_overlay
                overlay_red[:,:,3] = (img_overlay != 0)*overlay_alpha*255
                self.display['overlay_fig'] = self.display['axes'].imshow(increase_saturation(overlay_red.astype(np.uint8)), alpha = 0.5, zorder=2, interpolation=None)
            self.display['text'] = self.display['axes'].text(0,0,str(round(self.display['pos'][0],2))+" ; "+ str(round(self.display['pos'][1],2)), color='black', horizontalalignment='left', verticalalignment='top',  transform= self.display['axes'].transAxes)
            plt.show()
        else:
            if self.gps_pos is None:
                img = np.nan*np.ones(shape)
            else:
                img, _ = self.extract_from_map(self.display['gps_pos']+self.attitude.apply(self.display['pos'] + center,True), self.attitude, shape, self.display['scale'])
            self.display['img'].set_data(increase_saturation(np.nan_to_num(img)))
            
            if not self.display['overlay'] is None:
                img_overlay = np.nan_to_num(self.display['overlay'].predict_image(self.display['gps_pos']+self.attitude.apply(self.display['pos'] + center,True), self.attitude, (int(np.ceil(shape[0]/self.display['scale'])), int(np.ceil(shape[1]/self.display['scale'])))))
                overlay_red = np.zeros((np.shape(img_overlay)[0], np.shape(img_overlay)[1], 4))
                overlay_red[:,:,0] = img_overlay
                overlay_red[:,:,3] = (img_overlay != 0)*overlay_alpha*255
                if self.display['overlay_fig'] is None:
                    self.display['overlay_fig'] = self.display['axes'].imshow(increase_saturation(overlay_red.astype(np.uint8)), alpha = 0.5, zorder=2)
                else:
                    self.display['overlay_fig'].set_data(increase_saturation(overlay_red.astype(np.uint8)))
            else:
                if not self.display['overlay_fig'] is None:
                    self.display['overlay_fig'].set_data(np.zeros((int(np.ceil(shape[0]/self.display['scale'])), int(np.ceil(shape[1]/self.display['scale'])),4)))
                self.display['overlay_fig'] = None

            self.display['text'].set_text(str(round(self.display['pos'][0],2))+" ; "+ str(round(self.display['pos'][1],2)))
            plt.draw()
            plt.pause(0.001)
        
    def extract_from_map(self, gps_pos, attitude, shape, scale=1):
        """ Return an image from the map for a given position and attitude and with a given shape """
        data_temp = RadarData(0, np.ones((int(np.ceil(shape[0]/scale)), int(np.ceil(shape[1]/scale)))), gps_pos, attitude, self.precision)
        img1, cov_img1, new_origin = self.build_partial_map(data_temp)

        P_start = self.attitude.apply(new_origin - gps_pos)[0:2]/self.precision
        R = rotation_proj(self.attitude, attitude).inv().as_dcm()[:2,:2]
        M2 = np.concatenate((R,R.dot(np.array([[P_start[0]],[P_start[1]]]))), axis = 1)

        M2 = scale*np.eye(2).dot(M2)
        img2 = cv2.warpAffine(img1, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        cov_img2 = cv2.warpAffine(cov_img1, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        return img2, cov_img2