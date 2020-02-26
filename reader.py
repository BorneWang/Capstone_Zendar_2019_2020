import h5py
import numpy as np
from PIL import Image
from copy import deepcopy
from data import RadarData
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj

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
    
    def load_heatmaps(self, t_ini=0, t_final=np.inf):
        """ Function load radar data magnitude from HDF5 between t_ini and"""
        hdf5 = h5py.File(self.src,'r+')
        
        aperture_gt = hdf5['radar']['broad01']
        aperture = hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys())
           
        t0 = float(times[0])       
        times = [times[i] for i in range(len(times)) if float(times[i])-t0>t_ini and float(times[i])-t0<t_final]
        prev_perc = -1
        for i, t in enumerate(times):
            if np.floor(i/(len(times)-1)*10) != prev_perc:
                print("Loading data: "+str(np.floor(i/(len(times)-1)*10)*10)+"%")
                prev_perc = np.floor(i/(len(times)-1)*10)
            heatmap = aperture[t][...];       
            if not np.sum(heatmap) == 0:
                gps_pos = np.array(list(aperture[t].attrs['POSITION'][0]))
                att = np.array(list(aperture[t].attrs['ATTITUDE'][0]))
                self.heatmaps[float(t)-float(times[0])] = RadarData(float(t), np.array(Image.fromarray(heatmap, 'L')), gps_pos, rot.from_quat(att))
        self.tracklog_translation = aperture.attrs['tracklog_translation']
            
        groundtruth = ("groundtruth" in aperture_gt)
        if groundtruth:
            print("Loading groundtruth")
            aperture_gt = hdf5['radar']['broad01']['groundtruth']
            self.groundtruth_translation = aperture_gt.attrs['tracklog_translation']
            self.groundtruth = dict()
            times_gt = list(aperture_gt.keys())
            times_gt = [times_gt[i] for i in range(len(times_gt)) if (t_ini <= float(times_gt[i])-t0 < t_final) or (i+1<len(times_gt) and t_ini <= float(times_gt[i+1])-t0 < t_final) or (i-1>=0 and t_ini <= float(times_gt[i-1])-t0 < t_final)]
            gt_att, gt_pos, gt_time = [], [], []
            for i, t in enumerate(times_gt):
                gt_pos.append(np.array(list(aperture_gt[t].attrs['POSITION'][0])))
                gt_att.append(rot.from_quat(np.array(list(aperture_gt[t].attrs['ATTITUDE'][0]))))
                gt_time.append(float(t)-float(times[0]))
                
            times = list(self.heatmaps.keys())
            slerp = Slerp(gt_time, rot.from_quat([r.as_quat() for r in gt_att]))
            gt_att = [rot.from_quat(q) for q in slerp(times).as_quat()]
            gt_pos = np.array([np.interp(times, gt_time, np.array(gt_pos)[:,i]) for i in range(0,3)]).T
            for i in range(len(times)):
                self.groundtruth[times[i]] = {'POSITION': gt_pos[i], 'ATTITUDE': gt_att[i]}
        
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
    
    def plot_evaluation(self):
        """ Evaluate the transformation given in data information compared to image analysis """ 
        times = self.get_timestamps()
        if hasattr(self,"groundtruth"):
            pos_error_gps = np.zeros(len(times)-1)
            pos_error_cv2 = np.zeros(len(times)-1)
            att_error_gps = np.zeros(len(times)-1)
            att_error_cv2 = np.zeros(len(times)-1)
        else:         
            pos_error = np.zeros(len(times)-1)
            att_error = np.zeros(len(times)-1)
            
        for i in range(1,len(times)):           
            trans_cv, rotation_cv = self.heatmaps[times[i]].image_transformation_from(self.heatmaps[times[i-1]])
            theta_cv = rotation_cv.as_euler('zxy')[0]
            
            theta_gps = rotation_proj(self.heatmaps[times[i-1]].attitude, self.heatmaps[times[i]].attitude).as_euler('zxy')[0]
            trans_gps = self.heatmaps[times[i]].earth2rbd(self.heatmaps[times[i]].gps_pos - self.heatmaps[times[i-1]].gps_pos)
            
            if hasattr(self,"groundtruth"):
                theta_gt = rotation_proj(self.groundtruth[times[i-1]]['ATTITUDE'], self.groundtruth[times[i]]['ATTITUDE']).as_euler('zxy')[0]
                trans_gt = self.groundtruth[times[i]]['ATTITUDE'].apply(self.groundtruth[times[i]]['POSITION'] - self.groundtruth[times[i-1]]['POSITION'])
                
                pos_error_gps[i-1] = np.sqrt((trans_gt - trans_gps).dot((trans_gt - trans_gps).T))
                att_error_gps[i-1] = abs(theta_gt - theta_gps)
                pos_error_cv2[i-1] = np.sqrt((trans_gt - trans_cv).dot((trans_gt - trans_cv).T))
                att_error_cv2[i-1] = abs(theta_gt - theta_cv)
            else:             
                pos_error[i-1] = np.sqrt((trans_gps - trans_cv).dot((trans_gps - trans_cv).T))
                att_error[i-1] = abs(theta_gps - theta_cv)
            
        plt.figure()
        if hasattr(self,"groundtruth"):
            print("Average GPS translation error (m): " + str(np.mean(pos_error_gps)))
            print("Average cv2 translation error (m): " + str(np.mean(pos_error_cv2)))
            plt.title("Square root error of GPS and CV2 translations with groundtruth")
            plt.plot(times[1:], pos_error_gps, label="GPS")
            plt.plot(times[1:], pos_error_cv2, label="CV2")
            plt.legend()
        else:
            plt.title("Square root error between GPS and CV2 translations")
            print("Average GPS translation error (m): " + np.mean(pos_error))
            plt.plot(times[1:], pos_error)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")

        plt.figure()
        if hasattr(self,"groundtruth"):
            print("Average GPS rotation error (deg): " + str(np.rad2deg(np.mean(att_error_gps))))
            print("Average cv2 rotation error (deg): " + str(np.rad2deg(np.mean(att_error_cv2))))
            plt.title("Error of GPS and CV2 rotations with groundtruth")
            plt.plot(times[1:], att_error_gps, label="GPS")
            plt.plot(times[1:], att_error_cv2, label="CV2")
            plt.legend()
        else:
            print("Average GPS rotation error (rad): " + str(np.rad2deg(np.mean(att_error))))
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
                out.append(np.linalg.norm(self.heatmaps[times[i+1]].gps_pos - self.heatmaps[times[i]].gps_pos)/(times[i+1]- times[i]))
            return out
        else:
            if t_final is None:
                return np.linalg.norm(self.heatmaps[self.get_timestamps(t_ini+0.1, t_final)].gps_pos - self.heatmaps[self.get_timestamps(t_ini, t_final)].gps_pos)/0.1
            else:
                times = self.get_timestamps(t_ini, t_final)
                out = []
                for i in range(len(times)-1):
                    out.append(np.linalg.norm(self.heatmaps[times[i+1]].gps_pos - self.heatmaps[times[i]].gps_pos)/(times[i+1]- times[i]))
                return out
          
    def get_groundtruth_pos(self,t_ini=None, t_final=None):
        """ Return groundtruth position for time between t_ini and t_final """
        
        assert hasattr(self,"groundtruth"), "No groundtruth loaded"
        
        if t_ini is None:
            times = self.get_timestamps()
            return np.array([self.groundtruth[t]['POSITION'] for t in times])
        else:     
            times = self.get_timestamps(t_ini, t_final)
            if t_final is None:
                return self.groundtruth[times]['POSITION']
            else:
                return np.array([self.groundtruth[t]['POSITION'] for t in times])

    def get_groundtruth_att(self,t_ini=None, t_final=None):
        """ Return groundtruth attitude for time between t_ini and t_final """
        
        assert hasattr(self,"groundtruth"), "No groundtruth loaded"
        
        if t_ini is None:
            times = self.get_timestamps()
            return [self.groundtruth[t]['ATTITUDE'] for t in times]
        else:     
            times = self.get_timestamps(t_ini, t_final)
            if t_final is None:
                return self.groundtruth[times]['ATTITUDE']
            else:
                return [self.groundtruth[t]['ATTITUDE'] for t in times]