import cv2
import pickle
import numpy as np
from os import path
from PIL import Image
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot

def merge_img(img1, img2, P1, P2):
    img = np.nan*np.ones(np.shape(img1))
    cov_img = np.nan*np.ones(np.shape(img1))
    for i in range(0, np.size(img1,0)):
        for j in range(0, np.size(img1,1)):
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
       
def check_transform(data, rotation, translation, name):
    """ Save an image to vizualize the calculated transformation (for test purpose) """
    translation = translation/0.04
    shape = (np.shape(data.img)[1], np.shape(data.img)[0])
    warp_matrix = np.concatenate(((rotation.as_dcm()[:2,:2]).T,np.array([[-translation[0]],[-translation[1]]])), axis = 1)
    Image.fromarray(cv2.warpAffine(data.img, warp_matrix, shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)).save(name);    
     
class RadarData:
    
    def __init__(self, ts, img, gps_pos, attitude, precision=0.04):
        self.id = ts
        self.precision = precision
        self.img = img # array image (0-255)
        self.gps_pos = gps_pos # 1x3 array
        self.attitude = attitude # scipy quaternion
        
    def get_img(self):
        """ Return an image of the map (unknown area are set to zero) """ 
        return Image.fromarray(np.nan_to_num(self.img).astype(np.uint8))
        
    def height(self):
        """ Return max y position of a pixel in image frame """
        return self.precision*(np.size(self.img,0)-1)
    
    def width(self):
        """ Return max x position of a pixel in image frame """
        return self.precision*(np.size(self.img,1)-1)
        
    def meters2indices(self, point):
        """ Give the position of a pixel according its position in image frame """
        x_I = int(round(point[0]/self.precision))
        y_I = int(round(point[1]/self.precision))
        return x_I, y_I
    
    def earth2rbd(self,pos, inverse=False):
        """ Change of frame from earth frame to right-backward-down """
        return self.attitude.apply(pos, inverse)
    
    def image_grid(self):
        """ give the position of each pixel in the image frame """
        x, y = np.meshgrid(np.linspace(0, self.width(), np.size(self.img,1)), np.linspace(0, self.height(), np.size(self.img,0)))
        return np.dstack((x,np.zeros(np.shape(x)),y))
        
    def earth_grid(self):
        """ give the position of each pixel in the earthframe """
        img_grid = self.image_grid()
        earth_grid = self.earth2rbd(img_grid, True) + self.gps_pos
        return np.reshape(earth_grid, np.shape(img_grid))
    
    def image_transformation_from(self, otherdata):
        """ Return the translation and the rotation based on the two radar images """
        if path.exists("cv2_transformations.pickle"):        
            cv2_transformations = open("cv2_transformations.pickle","rb")
            trans_dict = pickle.load(cv2_transformations)
            cv2_transformations.close()
        else:
            trans_dict = dict()
            
        if str(self.id)+"-"+str(otherdata.id) in trans_dict:
            translation, rotation = trans_dict[str(self.id)+"-"+str(otherdata.id)]
        else:
            #TODO: 3D transformation of image (to take into account pitch and roll changes)
            warp_mode = cv2.MOTION_EUCLIDEAN
            number_of_iterations = 5000;
            termination_eps = 1e-9;
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            (cc, warp_matrix) = cv2.findTransformECC (otherdata.img, self.img, warp_matrix, warp_mode, criteria)
            
            rot_matrix = np.array([[warp_matrix[0,0], warp_matrix[1,0], 0], [warp_matrix[0,1], warp_matrix[1,1], 0], [0,0,1]])
            translation = -self.precision*np.array([warp_matrix[0,2], warp_matrix[1,2], 0])
            rotation = rot.from_dcm(rot_matrix)
            cv2_transformations = open("cv2_transformations.pickle","wb")
            trans_dict[str(self.id)+"-"+str(otherdata.id)] = (translation, rotation)
            pickle.dump(trans_dict, cv2_transformations)
            cv2_transformations.close()  
            
        #TODO: just for test and vizualisation, could be removed()
        #check_transform(self, rotation, translation, 'radar1_1.png')
            
        return translation, rotation
    
    def image_position_from(self, otherdata):
        """ Return the actual position and attitude based on radar images comparison """
        translation, rotation = self.transformation_to(otherdata)
        
        gps_pos = otherdata.gps_pos + self.earth2rbd(translation,True)
        attitude = self.attitude*rotation
        return gps_pos, attitude
    
    def predict_image(self, gps_pos, attitude):
        """ Give the prediction of an observation in a different position based on actual radar image """
        #TODO: 3D transformation of image (to take into account pitch and roll changes)
        exp_rot_matrix = rot.as_dcm(self.attitude.inv()*attitude)[:2,:2]
        
        exp_trans = self.earth2rbd(gps_pos - self.gps_pos)[0:2]/self.precision
        
        shape = (np.shape(self.img)[1], np.shape(self.img)[0])
        warp_matrix = np.concatenate((exp_rot_matrix, np.array([[-exp_trans[0]],[-exp_trans[1]]])), axis = 1)
        predict_img = cv2.warpAffine(self.img, warp_matrix, shape, flags=cv2.INTER_LINEAR, borderValue = 0);
        
        mask = cv2.warpAffine(np.ones(np.shape(self.img)), warp_matrix, shape, flags=cv2.INTER_LINEAR, borderValue = 0);
        diff = mask - np.ones(np.shape(self.img))
        diff[diff != -1] = 0
        diff[diff == -1] = np.nan
        prediction = diff + predict_img

        #TODO: just for test and vizualisation, could be removed()
        #Image.fromarray(predict_img).save('radar2_2.png')

        return prediction
    
class Map(RadarData):
    
    def __init__(self, radardata):
        self.id = radardata.id
        self.precision = radardata.precision
        self.img = deepcopy(radardata.img) # array for image 
        self.gps_pos = deepcopy(radardata.gps_pos) # 1x3 array
        self.attitude = deepcopy(radardata.attitude) # scipy quaternion
        
        self.img_cov = 10
        self.covariance_map = self.img_cov*np.ones(np.shape(self.img))
    
    def set_initial_covariance(self, cov):
        """ Set the initial covariance of the probability map """
        self.img_cov = cov
        self.covariance_map = cov*np.ones(np.shape(self.img))
    
    def get_covariance_map(self):
        """ Return an image of the covariance map """
        return Image.fromarray(np.nan_to_num(self.img).astype(np.uint8))
    
    def add_data(self,otherdata):
        """ Add a radardata to the map by producing a bigger map with the fusion of the two """
        P1 = np.array([0,0])
        P2 = np.array([self.width(),0])
        P3 = np.array([self.width(),self.height()])
        P4 = np.array([0,self.height()])
        
        q = rot.from_dcm(np.block([[rot.as_dcm(self.attitude.inv()*otherdata.attitude)[:2,:2], np.zeros((2,1))],[np.zeros((1,2)), 1]]))
        P5 = self.earth2rbd(otherdata.gps_pos - self.gps_pos)[0:2]
        P6 = P5 + q.apply(np.array([otherdata.width(),0,0]))[0:2]
        P7 = P5 + q.apply(np.array([otherdata.width(),otherdata.height(),0]))[0:2]
        P8 = P5 + q.apply(np.array([0,otherdata.height(),0]))[0:2]
        
        new_origin = np.array([min(P1[0],P2[0],P3[0],P4[0],P5[0],P6[0],P7[0],P8[0]), min(P1[1],P2[1],P3[1],P4[1],P5[1],P6[1],P7[1],P8[1]), 0])
        x_length = max(P1[0],P2[0],P3[0],P4[0],P5[0],P6[0],P7[0],P8[0]) - new_origin[0]
        y_length = max(P1[1],P2[1],P3[1],P4[1],P5[1],P6[1],P7[1],P8[1]) - new_origin[1]
        
        v1 = (new_origin[0:2]/self.precision).astype(np.int)
        new_origin = np.append(v1*self.precision,0)
        new_gpspos = self.gps_pos + self.attitude.inv().apply(new_origin)
        shape = (int(np.ceil(y_length/self.precision)),int(np.ceil(x_length/self.precision)));
        
        img1 = np.nan*np.ones(shape)
        img1[-v1[1]:np.size(self.img,0)-v1[1], -v1[0]:np.size(self.img,1)-v1[0]] = self.img
        cov_img1 = np.nan*np.ones(shape)
        cov_img1[-v1[1]:np.size(self.covariance_map,0)-v1[1], -v1[0]:np.size(self.covariance_map,1)-v1[0]] = self.covariance_map
        
        v2 = (P5-new_origin[0:2])/self.precision
        M2 = np.concatenate((rot.as_dcm(self.attitude.inv()*otherdata.attitude)[:2,:2],np.array([[v2[0]],[v2[1]]])), axis = 1)
        img2 = cv2.warpAffine(otherdata.img, M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        cov_img2 = cv2.warpAffine(self.img_cov*np.ones(np.shape(otherdata.img)), M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0)
        mask = cv2.warpAffine(np.ones(np.shape(otherdata.img)), M2, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue = 0);
        diff = mask - np.ones(shape)
        diff[diff != 0] = np.nan
        img2 = diff + img2
        cov_img2 = diff + cov_img2

        self.img, self.covariance_map = merge_img(img1, img2, cov_img1, cov_img2)
        self.gps_pos = deepcopy(new_gpspos)
        return self.get_img()