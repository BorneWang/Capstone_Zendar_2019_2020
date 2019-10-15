import h5py
import numpy as np
from PIL import Image
from data import RadarData
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as rot

class Reader:
    
    def __init__(self, src):
        self.src = src
        self.heatmaps = dict()
        self.load_heatmaps()
        self.min_magnitude = 0;
        self.max_magnitude = 0;
    
    def load_heatmaps(self):
        """ Function load rada data magnitude from HDF5 """
        #TODO: preprocessing in HDF5 by Bowen
        hdf5 = h5py.File(self.src,'r+')
        aperture = hdf5['radar']['broad01']['aperture2D']
        self.min_magnitude = aperture.attrs['min_value']
        self.max_magnitude = aperture.attrs['max_value']
        times = list(aperture.keys())
        for i, t in enumerate(times):
            heatmap = aperture[t][...];
            gps_pos = np.array(list(aperture[t].attrs['POSITION'][0]))
            att = np.array(list(aperture[t].attrs['ATTITUDE'][0]))
            self.heatmaps[float(t)-float(times[0])] = RadarData(Image.fromarray(np.uint8((heatmap-self.min_magnitude)/(self.max_magnitude-self.min_magnitude)*255), 'L'), gps_pos, rot.from_quat([att[1],att[2],att[3],att[0]]))
        hdf5.close()
        
    def play_video(self, t_ini, t_final, grayscale = True):
        """ Play a video of radar images between t_ini and t_final """
        # TODO: look Animation because problem with PIL
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
     
    def find_timestamps(self, t_ini, t_final):
        """ Return a list of data timestamps between t_ini and t_final """
        times = self.heatmaps.keys.sort()
        selection = []
        for t in times:
            if t>=t_ini and t<=t_final:
                selection.append(t)
        return selection
    
    def get_radardata(self, t_ini, t_final=None):
        """ Return radar data for time between t_ini and t_final """
        if t_final is None:
            return self.heatmaps[t_ini]
        else:
            times = self.find_timestamps(t_ini, t_final)
            return np.array([self.heatmaps[t] for t in times])
            
    def get_img(self, t_ini, t_final=None):
        """ Return radar data image for time between t_ini and t_final """
        if t_final is None:
            return self.heatmaps[t_ini].img
        else:
            times = self.find_timestamps(t_ini, t_final)
            return np.array([self.heatmaps[t].img for t in times])