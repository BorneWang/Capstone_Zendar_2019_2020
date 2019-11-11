import h5py
import numpy as np
from PIL import Image
from copy import deepcopy
from data import RadarData
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as rot

class Reader:
    
    def __init__(self, src, t_ini=0, t_final=np.inf, option=None):
        self.src = src
        self.heatmaps = dict()
        self.min_magnitude = 0;
        self.max_magnitude = 0;
        self.tracklog_translation = np.zeros(3)
        
        self.load_heatmaps(t_ini, t_final, option)
    
    def __iter__(self):
        self.iter = 0
        self.prev_perc = -1
        return self
    
    def __next__(self):
        N = len(list(self.heatmaps.items()))
        if np.floor(self.iter/(N-1)*10) != self.prev_perc:
            print("Processing data: "+str(np.floor(self.iter/(N-1)*10)*10)+"%")
            self.prev_perc = np.floor(self.iter/(N-1)*10)      
        self.iter+=1 
        if self.iter<=N:
            return list(self.heatmaps.items())[self.iter-1]
        else:
            print("Data processed")
            raise StopIteration
    
    def __getitem__(self, key):
        reader = deepcopy(self)
        reader.heatmaps = {k:self.heatmaps[k] for k in list(self.heatmaps.keys())[key]}
        return reader
    
    def load_heatmaps(self, t_ini=0, t_final=np.inf, option=None):
        """ Function load radar data magnitude from HDF5 between t_ini and"""
        hdf5 = h5py.File(self.src,'r+')
        aperture = hdf5['radar']['broad01']['aperture2D']
        self.min_magnitude = aperture.attrs['min_value']
        self.max_magnitude = aperture.attrs['max_value']
        self.tracklog_translation = aperture.attrs['tracklog_translation']
        times = list(aperture.keys())
        times = [times[i] for i in range(len(times)) if float(times[i])-float(times[0])>t_ini and float(times[i])-float(times[0])<t_final]
        N = len(times)
        prev_perc = -1
        for i, t in enumerate(times):
            if np.floor(i/(N-1)*10) != prev_perc:
                print("Loading data: "+str(np.floor(i/(N-1)*10)*10)+"%")
                prev_perc = np.floor(i/(N-1)*10)
            heatmap = aperture[t][...];       
            if option == 'increase_contrast':
                heatmap = self.increase_contrast(heatmap)
            else:
                heatmap = np.uint8((heatmap-self.min_magnitude)/(self.max_magnitude-self.min_magnitude)*255)
            if not np.sum(heatmap) == 0:
                gps_pos = np.array(list(aperture[t].attrs['POSITION'][0]))
                att = np.array(list(aperture[t].attrs['ATTITUDE'][0]))
                self.heatmaps[round(float(t)-float(times[0]), 2)] = RadarData(float(t), np.array(Image.fromarray(heatmap, 'L')), gps_pos, rot.from_quat(att))
        hdf5.close()
        print("Data loaded")
    
    def increase_contrast(self, heatmap):
        """ Increase the contrast of the contrast of the image with gamma method, affine transformation and thresholding """
        gamma = 0.57
        lin_coeff = 3 # Change this number (4.0, 3.0, 2.0)
        aff_coeff = 60
        thres = 16.3
        heatmap = (np.exp(heatmap)-np.exp(self.min_magnitude))/(np.exp(self.max_magnitude)-np.exp(self.min_magnitude))
        heatmap = np.power(heatmap, gamma)*255
        row, col = heatmap.shape
        for i in range(row):
            for j in range(col):
                if heatmap[i,j] > thres:
                    heatmap[i,j] = lin_coeff*heatmap[i,j] + aff_coeff
        heatmap = heatmap.astype(np.uint8)
        return heatmap
        
    def play_video(self, t_ini=0, t_final=np.inf, grayscale = True):
        """ Play a video of radar images between t_ini and t_final """
        anim_running = True
        def onClick(event):
            nonlocal anim_running
            if anim_running:
                ani.event_source.stop()
                anim_running = False
            else:
                ani.event_source.start()
                anim_running = True
                    
        times = self.find_timestamps(t_ini, t_final)
        images = []
        
        fig = plt.figure()
        for t in times:
            plt.axis('off')
            if grayscale:              
                images.append([plt.imshow(Image.fromarray(self.heatmaps[t].img), cmap='gray', vmin=0, vmax=255), plt.text(0.5,0.5,str(t))])
            else:
                images.append([plt.imshow(Image.fromarray(self.heatmaps[t].img)), plt.text(0.6,0.5,str(t))])
        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = animation.ArtistAnimation(fig, images, interval=100, blit=False, repeat_delay=1000)
        return ani
     
    def find_timestamps(self, t_ini, t_final=None):
        """ Return a list of data timestamps between t_ini and t_final """
        times = list(self.heatmaps.keys())
        times.sort()
        if t_final is None:
            t_adj = times[min(range(len(times)), key = lambda i: abs(times[i]-t_ini))]
            return t_adj
        else:    
            return [t for t in times if t>=t_ini and t<=t_final]
    
    def plot_gps_evaluation(self):
        """ Evaluate the transformation given in data information compared to image analysis """ 
        times = self.find_timestamps(0, np.inf)
        pos_error = np.zeros(len(times)-1)
        att_error = np.zeros(len(times)-1)
        
        prev_data = self.heatmaps[times[0]]
        for i in range(1,len(times)):
            data = self.heatmaps[times[i]]
            
            trans_cv, rotation_cv = data.image_transformation_from(prev_data)
            theta_cv = rotation_cv.as_euler('xyz')[2]
                
            theta_gps = (prev_data.attitude.inv()*data.attitude).as_euler('xyz')[2]
            trans_gps = data.earth2rbd(data.gps_pos - data.gps_pos)
            
            pos_error[i-1] = np.sqrt((trans_gps - trans_cv).dot((trans_gps - trans_cv).T))
            att_error[i-1] = abs(theta_gps - theta_cv)
            
            prev_data = data
            
        plt.figure()
        plt.title("Square root error between GPS and CV2 translations")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.plot(times[1:], pos_error)
        plt.figure()
        plt.title("Error between GPS and CV2 rotations")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (rad)")
        plt.plot(times[1:], att_error)
    
    def tracklog_translate(self, gps_positions, attitudes):
        """ Convert the position of the top left corner of image to car position """
        car_pos = []
        for i in range(len(gps_positions)):
            car_pos.append(attitudes[i].apply(attitudes[i].apply(gps_positions[i]) - self.tracklog_translation, True))
        return np.array(car_pos)
    
    def get_radardata(self, t_ini, t_final=None):
        """ Return radar data for time between t_ini and t_final """
        times = self.find_timestamps(t_ini, t_final)
        if t_final is None:
            return self.heatmaps[times]
        else:
            times = self.find_timestamps(t_ini, t_final)
            return np.array([self.heatmaps[t] for t in times])
            
    def get_img(self, t_ini, t_final=None):
        """ Return radar data image for time between t_ini and t_final """
        times = self.find_timestamps(t_ini, t_final)
        if t_final is None:
            return Image.fromarray(self.heatmaps[times].img)
        else:
            return np.array([Image.fromarray(self.heatmaps[t].img) for t in times])
        
    def get_gps_pos(self,t_ini, t_final=None):
        """ Return GPS position for time between t_ini and t_final """
        times = self.find_timestamps(t_ini, t_final)
        if t_final is None:
            return self.heatmaps[times].gps_pos
        else:
            return np.array([self.heatmaps[t].gps_pos for t in times])
        
    def get_gps_att(self,t_ini, t_final=None):
        """ Return GPS attitude for time between t_ini and t_final """
        times = self.find_timestamps(t_ini, t_final)
        if t_final is None:
            return self.heatmaps[times].attitude
        else:
            return [self.heatmaps[t].attitude for t in times]