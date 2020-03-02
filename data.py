import cv2
import pickle
import numpy as np
from os import path
from PIL import Image
from scipy.spatial.transform import Rotation as rot
 
from utils import check_transform, rotation_proj

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
            print("Calculating transformation")
         
            # ECC
            warp_mode = cv2.MOTION_EUCLIDEAN
            number_of_iterations = 625;
            termination_eps = 1e-9;
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            (cc, warp_matrix) = cv2.findTransformECC (otherdata.img.astype(np.uint8), self.img.astype(np.uint8), warp_matrix, warp_mode, criteria)

            # SIFT
            # warp_matrix = cv2.estimateRigidTransform(otherdata.img.astype(np.uint8), self.img.astype(np.uint8), False)
            
            rot_matrix = np.array([[warp_matrix[0,0], warp_matrix[1,0], 0], [warp_matrix[0,1], warp_matrix[1,1], 0], [0,0,1]])
            translation = -self.precision*np.array([warp_matrix[0,2], warp_matrix[1,2], 0])
            rotation = rot.from_dcm(rot_matrix)
            
            if not (otherdata.id == -1 or self.id == -1):      
                cv2_transformations = open("cv2_transformations.pickle","wb")
                trans_dict[str(self.id)+"-"+str(otherdata.id)] = (translation, rotation)
                pickle.dump(trans_dict, cv2_transformations)
                cv2_transformations.close()  
            
        # just for test and vizualisation, could be removed()
        # check_transform(self, rotation, translation, 'radar1_1.png')
            
        return translation, rotation
    
    def image_position_from(self, otherdata):
        """ Return the actual position and attitude based on radar images comparison """
        translation, rotation = self.image_transformation_from(otherdata)
        
        gps_pos = otherdata.gps_pos + otherdata.earth2rbd(translation,True)
        # TODO: check rotation application
        attitude = rotation.inv()*otherdata.attitude
        return gps_pos, attitude
    
    def predict_image(self, gps_pos, attitude):
        """ Give the prediction of an observation in a different position based on actual radar image """
        exp_rot_matrix = rotation_proj(self.attitude, attitude).as_dcm()[:2,:2]
        
        exp_trans = self.earth2rbd(gps_pos - self.gps_pos)[0:2]/self.precision
        
        shape = (np.shape(self.img)[1], np.shape(self.img)[0])
        warp_matrix = np.concatenate((exp_rot_matrix, np.array([[-exp_trans[0]],[-exp_trans[1]]])), axis = 1)
        predict_img = cv2.warpAffine(self.img, warp_matrix, shape, flags=cv2.INTER_LINEAR, borderValue = 0);
        
        mask = cv2.warpAffine(np.ones(np.shape(self.img)), warp_matrix, shape, flags=cv2.INTER_LINEAR, borderValue = 0);
        diff = mask - np.ones(np.shape(self.img))
        diff[diff != -1] = 0
        diff[diff == -1] = np.nan
        prediction = diff + predict_img

        # just for test and vizualisation, could be removed()
        # Image.fromarray(predict_img).save('radar2_2.png')

        return prediction
    
    def image_overlap(self,data2):
        """ Return only the image intersection """
        w1 = np.ones(np.shape(self.img))
        w2 = np.ones(np.shape(data2.img))
    
        white_1 = RadarData(0,w1,self.gps_pos,self.attitude)
        white_2 = RadarData(0,w2,data2.gps_pos,data2.attitude)
    
        mask1 = white_2.predict_image(self.gps_pos,self.attitude)
        mask2 = white_1.predict_image(data2.gps_pos,data2.attitude)
    
        out1 = np.multiply(mask1, self.img)
        out2 = np.multiply(mask2, data2.img)
        return out1, out2
