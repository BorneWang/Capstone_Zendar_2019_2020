import h5py
import numpy as np
from PIL import Image
from data import RadarData
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as rot

class Reader:
    
    def __init__(self, src, option=None):
        self.src = src
        self.heatmaps = dict()
        self.load_heatmaps(option)
        self.min_magnitude = 0;
        self.max_magnitude = 0;
    
    def unlog(self,image):
        return np.exp(image)
    
    def combine_method(self,image):
        img_255 = np.uint8(255*(image-self.min_magnitude)/(self.max_magnitude-self.min_magnitude))
        fi = img_255 / 255.0
        gamma = 0.57
        out = np.power(fi, gamma)
        img_gamma = 255.0*out
        img_linear = 3.0 * img_gamma # Change this number (4.0, 3.0, 2.0)
        img_linear[img_linear>255] = 255
        img_linear = np.around(img_linear)
        img_linear = img_linear.astype(np.uint8)
        return img_linear
        
    def increase_contrast(self,image):
        img = self.unlog(image)
        return self.combine_method(img)
    
    def load_heatmaps(self, option):
        """ Function load rada data magnitude from HDF5 """
        #TODO: preprocessing in HDF5 by Bowen
        hdf5 = h5py.File(self.src,'r+')
        aperture = hdf5['radar']['broad01']['aperture2D']
        self.min_magnitude = aperture.attrs['min_value']
        self.max_magnitude = aperture.attrs['max_value']
        times = list(aperture.keys())
        N = len(times)
        prev_perc = -1
        if option == 'increase_contrast':
            for i, t in enumerate(times):
                if np.floor(float(i)*10/N) != prev_perc:
                    print("Loading data: "+str(np.floor(float(i)*100/N))+"%")
                    prev_perc = round(float(i)*10/N)
                heatmap = aperture[t][...];
                heatmap = self.increase_contrast(heatmap)
                gps_pos = np.array(list(aperture[t].attrs['POSITION'][0]))
                att = np.array(list(aperture[t].attrs['ATTITUDE'][0]))
                self.heatmaps[round(float(t)-float(times[0]), 2)] = RadarData(Image.fromarray(heatmap, 'L'), gps_pos, rot.from_quat(att))
        else:
            for i, t in enumerate(times):
                if np.floor(float(i)*10/N) != prev_perc:
                    print("Loading data: "+str(np.floor(float(i)*100/N))+"%")
                    prev_perc = round(float(i)*10/N)
                heatmap = aperture[t][...];
                gps_pos = np.array(list(aperture[t].attrs['POSITION'][0]))
                att = np.array(list(aperture[t].attrs['ATTITUDE'][0]))
                self.heatmaps[round(float(t)-float(times[0]), 2)] = RadarData(Image.fromarray(heatmap, 'L'), gps_pos, rot.from_quat(att))  
        hdf5.close()
        print("Data loaded")
        
    def play_video(self, t_ini, t_final, grayscale = True):
        """ Play a video of radar images between t_ini and t_final """
        times = self.find_timestamps(t_ini, t_final)
        images = []
        for t in times:
            plt.axis('off')
            if grayscale:              
                images.append(plt.imshow(self.heatmaps[t].img, cmap='gray', vmin=0, vmax=255))
            else:
                images.append(plt.imshow(self.heatmaps[t].img))
        fig = plt.figure()
        ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=1000)
        plt.show()
        return ani
     
    def find_timestamps(self, t_ini, t_final=None):
        """ Return a list of data timestamps between t_ini and t_final """
        times = list(self.heatmaps.keys())
        times.sort()
        if t_final is None:
            t_adj = times[min(range(len(times)), key = lambda i: abs(times[i]-t_ini))]
            if t_adj != t_ini:       
                print(t_adj)
            return t_adj
        else:    
            selection = []
            for t in times:
                if t>=t_ini and t<=t_final:
                    selection.append(t)
            return selection
    
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
            return self.heatmaps[times].img
        else:
            return np.array([self.heatmaps[t].img for t in times])
        
    def get_gps_pos(self,t_ini, t_final=None):
        """ Return GPS pos for time between t_ini and t_final """
        times = self.find_timestamps(t_ini, t_final)
        if t_final is None:
            return self.heatmaps[times].gps_pos
        else:
            return np.array([self.heatmaps[t].gps_pos for t in times])
