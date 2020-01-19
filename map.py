import cv2
import h5py
import datetime
import numpy as np
from copy import deepcopy
from data import RadarData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot

def merge_img(img1, img2, P1, P2, P_start, P_end):
    """ Merge two images pixel by pixel, weighted by uncertainty, only in modified area """
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
    
    def __init__(self, name = None):
        self.chunk_size = 1000
        self.img_cov = 10
        self.display ={'text': "0 ; 0", 'img': None, 'scale': 1, 'pos': np.array([0,0,0]), 'axes': None, 'fig':None, 'gps_pos':None} 
        
        if name is None:
            # Create a black map
            self.map_name = 'map_'+str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")
            print("Creating map: map_"+str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")+'.h5')
            hdf5 = h5py.File('maps/'+self.map_name+'.h5','a')
            map_hdf5 = hdf5.create_group("map")
            cov_map_hdf5 = hdf5.create_group("covariance")
            ini_map = np.nan*np.ones((self.chunk_size, self.chunk_size))
            ini_cov = np.nan*np.ones((self.chunk_size, self.chunk_size))
            map_hdf5.create_dataset("0/0", data = ini_map, shape=(self.chunk_size, self.chunk_size) )        
            cov_map_hdf5.create_dataset("0/0", data = ini_cov, shape=(self.chunk_size, self.chunk_size) ) 
            self.precision = None
            self.gps_pos = None
            self.attitude = None
            hdf5.close()
        else:     
            # Retrieve an already existing map by name
            self.map_name = name
            hdf5 = h5py.File('maps/'+self.map_name+'.h5','r')
            map_hdf5 = hdf5["map"]
            self.precision = map_hdf5.attrs["PRECISION"]
            self.gps_pos = map_hdf5.attrs["POSITION"]
            self.attitude = rot.from_quat(map_hdf5.attrs["ATTITUDE"])
            hdf5.close()

    def init_map(self, radardata):
        """ Initialize the map with an initial radardata """
        hdf5 = h5py.File('maps/'+self.map_name+'.h5','a')
        map_hdf5 = hdf5["map"]
        
        map_hdf5.attrs["POSITION"] = radardata.gps_pos
        map_hdf5.attrs["ATTITUDE"] = radardata.attitude.as_quat()
        map_hdf5.attrs["PRECISION"] = radardata.precision
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
                
    def add_data(self,otherdata):
        """ Fusionning a new radardata with part of the map """
        if self.gps_pos is None:
            self.init_map(otherdata)
        
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
        
    def show(self, gps_pos = None):
        """ Show a matplotlib representation of the map """
        # Parameter for the map display
        speed_trans = 20
        speed_scroll = 0.1
        shape = (1000, 2000)
        
        if gps_pos is None:
            self.display['gps_pos'] = deepcopy(self.gps_pos)
        else:
            self.display['gps_pos'] = deepcopy(gps_pos)
        
        def press(event):
            if event.key == 'left':
                self.display['pos'] = self.display['pos'] - np.array([speed_trans*self.precision,0,0])
            elif event.key == 'right':
                self.display['pos'] = self.display['pos'] + np.array([speed_trans*self.precision,0,0])
            elif event.key == 'up':
                self.display['pos'] = self.display['pos'] - np.array([0,speed_trans*self.precision,0])
            elif event.key == 'down':               
                self.display['pos'] = self.display['pos'] + np.array([0,speed_trans*self.precision,0])
            img, _ = self.extract_from_map(self.display['gps_pos']+self.attitude.apply(self.display['pos'],True), self.attitude, shape, self.display['scale'])
            self.display['img'].set_data(np.nan_to_num(img))
            self.display['text'].set_text(str(round(self.display['pos'][0],2))+" ; "+ str(round(self.display['pos'][1],2)))
            plt.draw()
            
        def scroll(event):
            if event.step != 0:
                self.display['scale'] = self.display['scale'] + speed_scroll*event.step
            img, _ = self.extract_from_map(self.display['gps_pos']+self.attitude.apply(self.display['pos'],True), self.attitude, shape, self.display['scale'])
            self.display['img'].set_data(np.nan_to_num(img))
            plt.draw()
                 
        missing = (self.display['fig'] is None)
        if missing: 
            self.display['text'] = "0 ; 0"
            self.display['scale'] = 1
            self.display['pos'] = np.array([0,0,0])
            self.display['fig'] = plt.figure(num=self.map_name, facecolor=(1,1,1))
            self.display['fig'].canvas.mpl_connect('key_press_event', press)
            self.display['fig'].canvas.mpl_connect('scroll_event', scroll)
            self.display['axes'] = plt.axes()
            self.display['axes'].set_facecolor("black")
            self.display['axes'].get_xaxis().set_visible(False)
            self.display['axes'].get_yaxis().set_visible(False)
            if self.gps_pos is None:
                img = np.nan*np.ones(shape)
            else:
                img, _ = self.extract_from_map(self.display['gps_pos'], self.attitude, shape, self.display['scale'])
            self.display['img'] = self.display['axes'].imshow(np.nan_to_num(img), cmap='gray', vmin=0, vmax=255)
            self.display['text'] = self.display['axes'].text(0,0,str(round(self.display['pos'][0],2))+" ; "+ str(round(self.display['pos'][1],2)), color='black', horizontalalignment='left', verticalalignment='top',  transform= self.display['axes'].transAxes)
            plt.show()
        else:
            if self.gps_pos is None:
                img = np.nan*np.ones(shape)
            else:
                img, _ = self.extract_from_map(self.display['gps_pos']+self.attitude.apply(self.display['pos'],True), self.attitude, shape, self.display['scale'])
            self.display['img'].set_data(np.nan_to_num(img))
            self.display['text'].set_text(str(round(self.display['pos'][0],2))+" ; "+ str(round(self.display['pos'][1],2)))
            plt.draw()
            plt.pause(0.001)
        
    def extract_from_map(self, gps_pos, attitude, shape, scale=1):
        """ Return an image from the map for a given position and attitude and with a given shape """
        data_temp = RadarData(0, np.ones((int(np.ceil(shape[0]/scale)), int(np.ceil(shape[1]/scale)))), gps_pos, attitude, self.precision)
        img1, cov_img1, new_origin, _, _ = self.build_partial_map(data_temp)

        P_start = self.attitude.apply(new_origin - gps_pos)[0:2]/self.precision
        M2 = np.concatenate((rot.as_dcm(attitude.inv()*self.attitude)[:2,:2],np.array([[P_start[0]],[P_start[1]]])), axis = 1)
        M2 = scale*np.eye(2).dot(M2)
        img2 = cv2.warpAffine(img1, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        cov_img2 = cv2.warpAffine(cov_img1, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        return img2, cov_img2