import numpy as np
import h5py
from PIL import Image
from data import RadarData
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calculate_norm_log(image):
    """ Calculate the log of the norm of an image with complex pixels """
    row, col = image.shape
    new_image = np.zeros((row,col))
    for i in range(row):
        new_image[i,:] = list(map(lambda x: np.linalg.norm((x['real'],x['imag'])), image[i,:]))
    log_image = np.log(new_image)
    return log_image

class Reader:
    
    def __init__(self, src):
        self.src = src
        self.heatmaps = dict()
        self.load_heatmaps()
    
    def load_heatmaps(self):
        """ Function to convert data to magnitude in HDF5 """
        hdf5 = h5py.File(self.src,'r+')
        aperture = hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys())
        maxi = 0
        for i, t in enumerate(times):
            heatmap = calculate_norm_log(aperture[t])
            aperture[t][...] = heatmap
#            self.heatmaps[float(t)] = RadarData(Image.fromarray(int(aperture[t]['real']*255), 'L'), aperture[t].attrs['POSITION'], aperture[t].attrs['ATTITUDE'])
#           Image.fromarray(np.uint8(heatmap/np.max(heatmap)*255), 'L')           
            if np.max(heatmap) > maxi:
                maxi = np.max(heatmap)
            print(str(i)+'/4000')
        for t in enumerate(times):
            aperture[t][...] = aperture[t][...]/maxi*255
        hdf5.close()
        
    def play_video(self, t_ini, t_final):
        """ Play a video of radar images between t_ini and t_final """
        times = self.find_timestamps(t_ini, t_final)
        images = []
        for t in times:
            images.push(self.heatmaps[t].img)
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
                selection.push(t)
        return selection
    
    def get_radardata(self, t_ini, t_final=None):
        """ Return radar data for time between t_ini and t_final """
        if t_final is None:
            return self.heatmaps[t_ini]
        else:
            times = self.find_timestamps(t_ini, t_final)
            return np.array([self.heatmaps[t] for t in times])
            
    def get_heatmap_img(self, t_ini, t_final=None):
        """ Return radar data image for time between t_ini and t_final """
        if t_final is None:
            return self.heatmaps[t_ini].img
        else:
            times = self.find_timestamps(t_ini, t_final)
            return np.array([self.heatmaps[t].img for t in times])