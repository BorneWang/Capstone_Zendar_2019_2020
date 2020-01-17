import h5py
import numpy as np
from PIL import Image
from copy import deepcopy
from data import RadarData
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as rot

class Reader:
    
    def __init__(self, src, t_ini=0, t_final=np.inf):
        self.src = src
        self.heatmaps = dict()
        self.tracklog_translation = np.zeros(3)
        
        self.load_heatmaps(t_ini, t_final)
    
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
            if not np.sum(heatmap) == 0:
                gps_pos = np.array(list(aperture[t].attrs['POSITION'][0]))
                att = np.array(list(aperture[t].attrs['ATTITUDE'][0]))
                self.heatmaps[float(t)-float(times[0])] = RadarData(float(t), np.array(Image.fromarray(heatmap, 'L')), gps_pos, rot.from_quat(att))
        
        aperture = hdf5['radar']['broad01']
        if "groundtruth" in aperture:
            aperture = hdf5['radar']['broad01']['groundtruth']
            self.groundtruth_translation = aperture.attrs['tracklog_translation']
            self.groundtruth = {"POSITION":[], "ATTITUDE":[], "TIME":[]}
            times = list(aperture.keys())
            times = [times[i] for i in range(len(times)) if float(times[i])-float(times[0])>=t_ini and float(times[i])-float(times[0])<t_final]
            for i, t in enumerate(times):
                self.groundtruth["POSITION"].append(np.array(list(aperture[t].attrs['POSITION'][0])))
                self.groundtruth["ATTITUDE"].append(rot.from_quat(np.array(list(aperture[t].attrs['ATTITUDE'][0]))))
                self.groundtruth["TIME"].append(float(t)-float(times[0]))
        
        hdf5.close()
        print("Data loaded")
        
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
                    
        times = self.get_timestamps(t_ini, t_final)
        images = []
        
        fig = plt.figure()
        for t in times:
            plt.axis('off')
            if grayscale:              
                images.append([plt.imshow(Image.fromarray(self.heatmaps[t].img), cmap='gray', vmin=0, vmax=255), plt.text(0.5,0.5,str(round(t,2)))])
            else:
                images.append([plt.imshow(Image.fromarray(self.heatmaps[t].img)), plt.text(0.6,0.5,str(round(t,2)))])
        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = animation.ArtistAnimation(fig, images, interval=100, blit=False, repeat_delay=1000)
        return ani
     
    def get_timestamps(self, t_ini=None, t_final=None):
        """ Return a list of data timestamps between t_ini and t_final """
        times = list(self.heatmaps.keys())
        if (t_ini is None) or (t_ini == 0 and t_final==np.inf):
            return times
        else:
            times.sort()
            if t_final is None:
                t_adj = times[min(range(len(times)), key = lambda i: abs(times[i]-t_ini))]
                return t_adj
            else:    
                return [t for t in times if t>=t_ini and t<=t_final]
    
    def plot_gps_evaluation(self):
        """ Evaluate the transformation given in data information compared to image analysis """ 
        times = self.get_timestamps()
        while times[-1]>self.groundtruth["TIME"][-1]:
            times = np.delete(times, len(times)-1)
        if hasattr(self,"groundtruth"):
            slerp = Slerp(self.groundtruth["TIME"], rot.from_quat([r.as_quat() for r in self.groundtruth["ATTITUDE"]]))
            gt_att = [rot.from_quat(q) for q in slerp(list(self.heatmaps.keys())).as_quat()]
            gt_pos = np.array([np.interp(list(self.heatmaps.keys()), self.groundtruth["TIME"], np.array(self.groundtruth["POSITION"])[:,i]) for i in range(0,3)]).T
        
            pos_error_gps = np.zeros(len(times)-1)
            pos_error_cv2 = np.zeros(len(times)-1)
            att_error_gps = np.zeros(len(times)-1)
            att_error_cv2 = np.zeros(len(times)-1)
        else:         
            pos_error = np.zeros(len(times)-1)
            att_error = np.zeros(len(times)-1)
            
        prev_data = self.heatmaps[times[0]]
        for i in range(1,len(times)):
            data = self.heatmaps[times[i]]
            
            trans_cv, rotation_cv = data.image_transformation_from(prev_data)
            theta_cv = rotation_cv.as_euler('zxy')[0]
            
            theta_gps = (prev_data.attitude.inv()*data.attitude).as_euler('zxy')[0]
            trans_gps = data.earth2rbd(data.gps_pos - prev_data.gps_pos)
            
            if hasattr(self,"groundtruth"):
                theta_gt = (gt_att[i-1].inv()*gt_att[i]).as_euler('zxy')[0]
                trans_gt = data.earth2rbd(gt_pos[i] - gt_pos[i-1])
                
                pos_error_gps[i-1] = np.sqrt((trans_gt - trans_gps).dot((trans_gt - trans_gps).T))
                att_error_gps[i-1] = abs(theta_gt - theta_gps)
                pos_error_cv2[i-1] = np.sqrt((trans_gt - trans_cv).dot((trans_gt - trans_cv).T))
                att_error_cv2[i-1] = abs(theta_gt - theta_cv)
            else:             
                pos_error[i-1] = np.sqrt((trans_gps - trans_cv).dot((trans_gps - trans_cv).T))
                att_error[i-1] = abs(theta_gps - theta_cv)
            
            prev_data = data
            
        plt.figure()
        if hasattr(self,"groundtruth"):
            plt.title("Square root error of GPS and CV2 translations with groundtruth")
            plt.plot(times[1:], pos_error_gps, label="GPS")
            plt.plot(times[1:], pos_error_cv2, label="CV2")
            plt.legend()
        else:
            plt.title("Square root error between GPS and CV2 translations")
            plt.plot(times[1:], pos_error)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")

        plt.figure()
        if hasattr(self,"groundtruth"):
            plt.title("Error of GPS and CV2 rotations with groundtruth")
            plt.plot(times[1:], att_error_gps, label="GPS")
            plt.plot(times[1:], att_error_cv2, label="CV2")
            plt.legend()
        else:
            plt.title("Error between GPS and CV2 rotations")
            plt.plot(times[1:], att_error)   
        plt.xlabel("Time (s)")
        plt.ylabel("Error (rad)")
    
    def get_radardata(self, t_ini=None, t_final=None):
        """ Return radar data for time between t_ini and t_final """
        if t_ini is None:
            times = self.get_timestamps()
            return np.array([self.heatmaps[t] for t in times])
        else:
            times = self.get_timestamps(t_ini, t_final)
            if t_final is None:
                return self.heatmaps[times]
            else:
                return np.array([self.heatmaps[t] for t in times])
            
    def get_img(self, t_ini=None, t_final=None):
        """ Return radar data image for time between t_ini and t_final """
        if t_ini is None:
            times = self.get_timestamps()
            return np.array([Image.fromarray(self.heatmaps[t].img) for t in times])
        else:
            times = self.get_timestamps(t_ini, t_final)
            if t_final is None:
                return Image.fromarray(self.heatmaps[times].img)
            else:
                return np.array([Image.fromarray(self.heatmaps[t].img) for t in times])
        
    def get_gps_pos(self,t_ini=None, t_final=None):
        """ Return GPS position for time between t_ini and t_final """
        if t_ini is None:
            times = self.get_timestamps()
            return np.array([self.heatmaps[t].gps_pos for t in times])
        else:     
            times = self.get_timestamps(t_ini, t_final)
            if t_final is None:
                return self.heatmaps[times].gps_pos
            else:
                return np.array([self.heatmaps[t].gps_pos for t in times])
        
    def get_gps_att(self,t_ini=None, t_final=None):
        """ Return GPS attitude for time between t_ini and t_final """
        if t_ini is None:
            times = self.get_timestamps()
            return [self.heatmaps[t].attitude for t in times]
        else:
            times = self.get_timestamps(t_ini, t_final)
            if t_final is None:
                return self.heatmaps[times].attitude
            else:
                return [self.heatmaps[t].attitude for t in times]
        
    def get_gps_speed(self, t_ini=None, t_final=None):
        """ Return GPS speed for time between t_ini and t_final """
        if t_ini is None:
            times = self.get_timestamps()
            out = []
            for i in range(len(times)-1):
                out.append(np.linalg.norm(self.heatmaps[times[i+1]].gps_pos - self.heatmaps[times[i+1]].gps_pos))
            return out
        else:
            if t_final is None:
                return np.linalg.norm(self.heatmaps[self.get_timestamps(t_ini+0.1, t_final)].gps_pos - self.heatmaps[self.get_timestamps(t_ini, t_final)].gps_pos)
            else:
                times = self.get_timestamps(t_ini, t_final)
                out = []
                for i in range(len(times)-1):
                    out.append(np.linalg.norm(self.heatmaps[times[i+1]].gps_pos - self.heatmaps[times[i+1]].gps_pos))
                return out
        