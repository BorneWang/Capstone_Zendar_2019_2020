import cv2
import h5py
import datetime
import numpy as np
from copy import deepcopy
from data import RadarData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot

def merge_img(img1, img2, P1, P2, P_start, P_end):
    img = deepcopy(img1)
    cov_img = deepcopy(P1)
    for i in range(max(0,P_start[1]), min(P_end[1], np.size(img1, 0))):
        for j in range(max(0,P_start[0]), min(P_end[0], np.size(img1, 1))):
            if np.isnan(img1[i][j]):
                img[i][j] = deepcopy(img2[i][j])
                cov_img[i][j] = deepcopy(P2[i][j])
            elif np.isnan(img2[i][j]):
                img[i][j] = deepcopy(img1[i][j])
                cov_img[i][j] = deepcopy(P1[i][j])
            else:
                img[i][j] = round((img1[i][j]*P2[i][j] + img2[i][j]*P1[i][j])/(P1[i][j]+P2[i][j]))
                cov_img[i][j] = P1[i][j]*P2[i][j]/(P1[i][j] + P2[i][j])
    return img, cov_img

class Map():
    
    def __init__(self, radardata):
        self.chunk_size = 1000
        self.img_cov = 10
        
        if isinstance(radardata, str):
            self.map_name = 'maps/' + radardata
            hdf5 = h5py.File(self.map_name,'r')
            map_hdf5 = hdf5["map"]
            self.precision = map_hdf5.attrs["PRECISION"]
            self.gps_pos = map_hdf5.attrs["POSITION"]
            self.attitude = rot.from_quat(map_hdf5.attrs["ATTITUDE"])
            hdf5.close()
        else:        
            self.map_name = 'maps/map_'+str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")+'.h5'
            print("Creating map: map_"+str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")+'.h5')
            hdf5 = h5py.File(self.map_name,'a')
            map_hdf5 = hdf5.create_group("map")
            cov_map_hdf5 = hdf5.create_group("covariance")
            
            ini_map = np.nan*np.ones((self.chunk_size, self.chunk_size))
            ini_map[:np.shape(radardata.img)[0], :np.shape(radardata.img)[1]] = deepcopy(radardata.img)
            ini_cov = np.nan*np.ones((self.chunk_size, self.chunk_size))
            ini_cov[:np.shape(radardata.img)[0], :np.shape(radardata.img)[1]] = self.img_cov*np.ones(np.shape(radardata.img))
            map_hdf5.create_dataset("0/0", data = ini_map, shape=(self.chunk_size, self.chunk_size) )        
            cov_map_hdf5.create_dataset("0/0", data = ini_cov, shape=(self.chunk_size, self.chunk_size) ) 
            map_hdf5.attrs["POSITION"] = radardata.gps_pos
            map_hdf5.attrs["ATTITUDE"] = radardata.attitude.as_quat()
            map_hdf5.attrs["PRECISION"] = radardata.precision
            hdf5.close()
            
            self.precision = radardata.precision
            
            self.gps_pos = deepcopy(radardata.gps_pos) 
            self.attitude = deepcopy(radardata.attitude)

    def build_partial_map(self, otherdata):
        """ Build partial map of the chunks that are needed to contain the new data """
        hdf5 = h5py.File(self.map_name,'a')
        map_hdf5 = hdf5["map"]
        cov_map_hdf5 = hdf5["covariance"]
        
        q = rot.from_dcm(np.block([[rot.as_dcm(self.attitude.inv()*otherdata.attitude)[:2,:2], np.zeros((2,1))],[np.zeros((1,2)), 1]]))
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
        hdf5.close()
        return img, cov_img, gps_pos, P9, P10
    
    def update_map(self, img, cov_img, pos):
        hdf5 = h5py.File(self.map_name,'a')
        map_hdf5 = hdf5["map"]
        cov_map_hdf5 = hdf5["covariance"]
        
        chunk = np.flip(np.round((self.attitude.apply(pos - self.gps_pos)[0:2]/self.precision)/self.chunk_size).astype(np.int))
        for i in range(chunk[0], chunk[0]+int(np.ceil(np.shape(img)[0]/self.chunk_size))):
            for j in range(chunk[1], chunk[1]+int(np.ceil(np.shape(img)[1]/self.chunk_size))):
                map_hdf5[str(i)+"/"+str(j)][...] = img[(i-chunk[0])*self.chunk_size:(i-chunk[0]+1)*self.chunk_size, (j-chunk[1])*self.chunk_size:(j-chunk[1]+1)*self.chunk_size]
                cov_map_hdf5[str(i)+"/"+str(j)][...] = cov_img[(i-chunk[0])*self.chunk_size:(i-chunk[0]+1)*self.chunk_size, (j-chunk[1])*self.chunk_size:(j-chunk[1]+1)*self.chunk_size]
        hdf5.close()
                
    def add_data(self,otherdata):
        """ Add a radardata to the map """
        img1, cov_img1, new_origin, P9, P10 = self.build_partial_map(otherdata)
        
        shape = np.shape(img1)
        v2 = self.attitude.apply(otherdata.gps_pos - new_origin)[0:2]/self.precision
        M2 = np.concatenate((rot.as_dcm(self.attitude.inv()*otherdata.attitude)[:2,:2],np.array([[v2[0]],[v2[1]]])), axis = 1)
        img2 = cv2.warpAffine(otherdata.img, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        cov_img2 = cv2.warpAffine(self.img_cov*np.ones(np.shape(otherdata.img)), M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        mask = cv2.warpAffine(np.ones(np.shape(otherdata.img)), M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0);
        diff = mask - np.ones(shape)
        diff[diff != 0] = np.nan
        img2 = diff + img2
        cov_img2 = diff + cov_img2

        P_start = np.floor((P9 - self.attitude.apply(new_origin - self.gps_pos)[0:2])/self.precision).astype(np.int)
        P_end = np.ceil((P10 - self.attitude.apply(new_origin - self.gps_pos)[0:2])/self.precision).astype(np.int)

        img, cov_img = merge_img(img1, img2, cov_img1, cov_img2, P_start, P_end)
        self.update_map(img, cov_img, new_origin)
        
    def show(self):
        hdf5 = h5py.File(self.map_name,'a')
        map_hdf5 = hdf5["map"]
        
        pos_x = 0
        pos_y = 0   
        def press(event):
            nonlocal pos_x
            nonlocal pos_y
            nonlocal im
            nonlocal t
            hdf5 = h5py.File(self.map_name,'a')
            map_hdf5 = hdf5["map"]
            if event.key == 'left':
                if str(pos_y)+"/"+str(pos_x-1) in map_hdf5:
                    pos_x = pos_x -1
            elif event.key == 'right':
                if str(pos_y)+"/"+str(pos_x+1) in map_hdf5:
                    pos_x = pos_x +1
            elif event.key == 'up':
                if str(pos_y-1)+"/"+str(pos_x) in map_hdf5:
                    pos_y = pos_y -1
            elif event.key == 'down':               
                if str(pos_y+1)+"/"+str(pos_x) in map_hdf5:
                    pos_y = pos_y +1
            im.set_data(np.nan_to_num(map_hdf5[str(pos_y)+"/"+str(pos_x)]))
            t.set_text(str(pos_y)+"/"+str(pos_x))
            plt.draw()
            hdf5.close()
            
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', press)
        im = plt.imshow(np.nan_to_num(map_hdf5[str(pos_y)+"/"+str(pos_x)]), cmap='gray', vmin=0, vmax=255)
        t = plt.text(0.6,0.5,str(pos_y)+"/"+str(pos_x))
        plt.axis('off')
        plt.show()
        hdf5.close()
        
    def extract_from_map(self, gps_pos, attitude, shape):
        data_temp = RadarData(0, np.ones(shape), gps_pos, attitude.inv(), self.precision)
        img1, cov_img1, new_origin, P9, P10 = self.build_partial_map(data_temp)

        P_start = np.floor((P9 - self.attitude.apply(new_origin - gps_pos)[0:2])/self.precision).astype(np.int)
        M2 = np.concatenate((rot.as_dcm(attitude.inv()*self.attitude)[:2,:2],np.array([[-P_start[0]],[-P_start[1]]])), axis = 1)
        img2 = cv2.warpAffine(img1, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        cov_img2 = cv2.warpAffine(cov_img1, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        mask = cv2.warpAffine(np.ones(shape), M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0);
        diff = mask - np.ones(shape)
        diff[diff != 0] = np.nan
        img2 = diff + img2
        cov_img2 = diff + cov_img2
        return img2, cov_img2