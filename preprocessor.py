import h5py
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
from statistics import stdev

from utils import DBSCAN_filter, change_attributes_frame

class Preprocessor:
    
    def __init__(self, src, goal, groundtruth, mean = None, std = None):
        """ Class performing the preprocessing of the dataset
            parameters:
            src: source hdf5 file name
            goal: the name of the processed file
            groundtruth: groundtruth hdf5 file name, if there is no groundtruth
                          this should be None.
            mean: pre set a mean value for normalization stage (optional)
            std: pre set a std value for normalization stage (optional)
            
            Introduction:
            This Preprocessor is mainly used to process the 2-D complex number
            radar matrix data (the format should be hdf5). The output of this
            program is a new hdf5 file with the following structure:
            file:
              -'radar'
                  -'broad01'
                      -'aperture2D'
                          attrs: 
                              preprocessed = True
                              tracklog_translation = [x, y ,z]
                          preprocessed image data (np.array with unint8 dtype)
                              key: Timestamp
                              attrs:
                                  POSITION
                                  ATTITUDE
                                  TIMESTAMP_SPAN
                                  APERTURE_SPAN
                      -'groundtruth' (if groundtruth is not None)
                          attrs:
                              tracklog_translation = [x, y ,z]
                          Dataset
                              key: Timestamp
                              data: 0
                              attrs:
                                  POSITION
                                  ATTITUDE
              -'tracklog'
            
            Remark:
            There are some words in the file structure need to be explained in
            detail.
              1. preprocessed image data
              2. tracklog_translation
              3. POSITION
              4. ATTITUDE
              5. groundtruth
            
            1. preprocessed image data
              a) calculate the norm of the complex number
              b) do mirror to the matrix and rotate it 90 degree to make sure the
                  image is (500,750) shape and on the right of the car.
              c) do np.log on the pixel value of the matrix
              d) Go through all the images and calculate a global mean and std ofutilize
                  the dataset
              e) use the global mean and std to normalize the matrix to 0-255
                  and change the dtype to uint8
              f) apply Ostu Thresholding algorithm to the imamge and get a
                  binary mask.
              g) utilize DBSCAN(Density-Based Spatial Clustering of Applications 
                  with Noise) algorithm to do clustering on the mask
              h) apply the mask on the image and get the preprocessed image
            
            2. tracklog_translation
              The tracklog_translation is an average translation vector from the
              position in tracklog to the position in aperture2D with the same 
              timestamp.
              
            3. POSITION
              The position in the source file is actually the bottom right of the
              image. We transfer this position to the upper left of the image.
              All the positions here are ECEF position.
            
            4. ATTITUDE
              In order to make the attitude suitable for CV2, we did the following
              process:
                  a) transfer from (w,x,y,z) to (x,y,z,w)
                  b) do inverse
                  c) multiply [0,-1,0],[-1,0,0],[0,0,-1]
              The result new attitude is used to directly transfer ECEF position to
              our CV2 coordinate position (x->right, y->down)
            
            5.groundtruth
              The position and attitude are from SBG data. If there is no SBG data
              user can directly set the parameter "groundtruth" to None.
              And the POSITION and ATTITUDE here will be processed the same way above.
        """        
        # parameters that could be tuned for new datasets
        self.GaussianBlur_kernel = (9,9)
        self.GaussianBlur_scale = 0
        self.DBSCAN_eps = 5.0
        self.DBSCAN_min_samples = 30
        
        # load files
        self.f = h5py.File(src,'r')
        self.aperture = self.f['radar']['squint_left_facing']['aperture2D']
        self.keys = list(self.aperture.keys())
            
        # create write-in data
        self.f_new = h5py.File(goal,'w')
        self.aperture_new = self.f_new.create_group("radar").create_group("broad01").create_group("aperture2D")
        
        self.images = []     # set temp images list and temp data
        
        # options
        self.goal = goal
        self.gt = groundtruth
        self.mean = mean
        self.std = std
        
    def run(self):
        """ Run the preprocessing of the data set """
        self.magnitude()   # do magnitude (from complex to read)              
        self.normalization()    # do global normalization, DBSCAN filtering and copy attrs
        
        self.aperture_new.attrs.create('preprocessed', True)        # add flag
              
        tracklog1 = self.f['tracklog']                              # copy tracklog
        self.f_new.create_dataset('tracklog',data=tracklog1[...])
        self.tracklog_trans(self.goal)                              # get translation from tracklog to gps
        self.f.close()
        self.f_new.close()
        
        if self.gt is not None:
            self.adding_groundtruth()   # add groundtruth to the file
    
    def magnitude(self):
        """ Perform magnitude calculations and mirror rotation of each image """
        batch = 50                          # process images in 50-size batch
        ite = len(self.keys)//batch
        start = 0
        for i in list(np.linspace(batch,ite*batch,ite).astype('int')):
            tic = time.time()
            images = list(map(lambda x:self.do_norm_mirror_rotate(self.aperture[x]), self.keys[start:i]))
            for j in range(start,i):
                idx = j - start
                self.images.append(np.log(images[idx]))
                
            start = i
            toc = time.time()
            print("batch number:",i,"/",len(self.keys),"time:",toc-tic)
            
        # residual images
        images = list(map(lambda x:self.do_norm_mirror_rotate(self.aperture[x]), self.keys[start:len(self.keys)]))
        for j in range(start,len(self.keys)):
            idx = j - start
            self.images.append(np.log(images[idx]))

        print("total images:",j+1,"Finished!")
       
    def get_global_mean(self):
        """ Calculate the global mean of the dataset or return predefined mean to use """
        if not self.mean is None:
            global_mean = self.mean
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
            self.mean = global_mean
        return global_mean
       
    def get_global_std(self):
        """ Calculate the global std of the dataset or return predefined std to use """
        if not self.std is None:
            global_std = self.std
        else:
            img_count = 1
            sum_ = 0.0
            count = 0
            for img in self.images:
                row, col = img.shape
                for i in range(row):
                    for j in range(col):
                        if img[i][j] > 0:
                            sum_ += (img[i][j]-self.get_global_mean())**2
                            count += 1
                print("img count:", img_count)
                img_count += 1
            global_std = (sum_/count)**(0.5)
            self.std = global_std
            print("global_std is:", global_std)
        return global_std
        
    def normalization(self):
        """ Perform a global shifted normalization of the dataset """
        # check correct:
        if len(self.images) == len(self.keys):
            print("list length correct, continue")
        else:
            print("list length error, stop")
            return
        
        global_mean = self.get_global_mean()    # get global mean
        global_std = self.get_global_std()      # get global std       
        
        for i in range(len(self.keys)):
            heatmap = self.images[i]
            heatmap = ((heatmap-global_mean)/global_std)*255.0/4.0
            heatmap[heatmap < 0] = 0
            heatmap[heatmap > 255] = 255
            heatmap = heatmap.astype(np.uint8)           
            
            # DBSCAN filtering
            if self.DBSCAN:
                if np.nanmax(heatmap) > 0:
                    heatmap = DBSCAN_filter(heatmap, kernel=self.GaussianBlur_kernel, 
                                            scale=self.GaussianBlur_scale, 
                                            eps=self.DBSCAN_eps,
                                            min_samples=self.DBSCAN_min_samples, 
                                            binary=False)
                    print("DBSCAN procedure:", i)
                        
            self.image_new = self.aperture_new.create_dataset(self.keys[i],data=heatmap) #save
            self.adding_attrs(i)                                                         # add attrs
              
    def tracklog_trans(self, goal):
        """ Add the tracklog translation attribute to the new dataset """
        tklog = Tracklog(goal)
        self.aperture_new.attrs.create('tracklog_translation',tklog.translations_POV_mean)
        
    def adding_attrs(self, idx):
        """ Add attributes to the image idx (POSITION, ATTITUDE, TIMESTAMP_SPAN, APERTURE_SPAN) """
        old_image = self.aperture[self.keys[idx]]        
        new_quat, p_topleft_global = change_attributes_frame(old_image)     # processing position and attitude

        # save to h5
        self.image_new.attrs.create('ATTITUDE', np.array([(new_quat[0], new_quat[1],
                                              new_quat[2], new_quat[3])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('w', '<f8')]))
        self.image_new.attrs.create('POSITION', np.array([(p_topleft_global[0],p_topleft_global[1],
                                                  p_topleft_global[2])],dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]))
        
        # copy other attrs
        self.image_new.attrs.create('TIMESTAMP_SPAN',old_image.attrs['TIMESTAMP_SPAN'])
        self.image_new.attrs.create('APERTURE_SPAN',old_image.attrs['APERTURE_SPAN'])
        
    def calculate_norm(self, image):
        """ Calculate magnitude of given image """
        row, col = image.shape
        new_image = np.zeros((row,col))
        for i in range(row):
            new_image[i,:] = list(map(lambda x: np.linalg.norm((x['real'],x['imag'])), image[i,:]))            
        return new_image
    
    def calculate_mirror_rotate(self, image):
        """ Rotate and mirror the image """
        return np.rot90(np.fliplr(image),3)        
        
    def do_norm_mirror_rotate(self,image):
        return self.calculate_mirror_rotate(self.calculate_norm(image))
    
    def adding_groundtruth(self):
        # read data
        f1 = h5py.File(self.goal,'r+')
        f2 = h5py.File(self.gt,'r')        

        aperture2 = f2['radar']['squint_left_facing']['aperture2D']
        keys = list(aperture2.keys())

        broad1 = f1['radar']['broad01']
        groundtruth = broad1.create_group('groundtruth')

        for key in keys:
            img = aperture2[key]
            img_new = groundtruth.create_dataset(key,data=0)
            
            new_quat, p_topleft_global = change_attributes_frame(img)   # processing position and attitude
            
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
        self.load_data(src)
        self.get_translations(src)
        print("time consume:", time.time()-tic)
        
    def load_data(self, src):
        hdf5 = h5py.File(src,'r')                               # load h5 file
        aperture = hdf5['radar']['broad01'][self.foldername]    # radar image data
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
        hdf5 = h5py.File(src,'r')                                #load file
        aperture = hdf5['radar']['broad01'][self.foldername]     # radar image data
        times = list(aperture.keys())
        N_img = len(times)

        trans_list = []
        POV_x_list = []
        POV_y_list = []
        POV_z_list = []

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
        