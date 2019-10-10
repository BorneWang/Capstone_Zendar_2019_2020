from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as rot
import cv2

def check_transform(data, warp_matrix, name):
    """ Save an image to vizualize the calculated transformation (for test purpose) """
    Image.fromarray(cv2.warpAffine(np.array(data.img), warp_matrix, data.img.size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)).save(name);    

class RadarData:
    
    def __init__(self, img, gps_pos, attitude, precision=0.04):
        self.precision = precision
        self.img = img # PIL image  
        self.gps_pos = gps_pos # 1x3 array
        self.attitude = attitude # scipy quaternion
        
    def height(self):
        """ Return max y position of a pixel in image frame """
        return self.precision*(self.img.height-1)
    
    def width(self):
        """ Return max x position of a pixel in image frame """
        return self.precision*(self.img.width-1)
        
    def meters2indices(self, point):
        """ Give the position of a pixel according its position in image frame """
        x_I = int(round(point[0]/self.precision))
        y_I = int(round(point[1]/self.precision))
        return x_I, y_I
    
    def earth2flu(self,pos, inverse=False):
        """ Change of frame from earth frame to front-left-up """
        return self.attitude.apply(pos, inverse)
    
    def image_grid(self):
        """ give the position of each pixel in the image frame """
        x, y = np.meshgrid(np.linspace(0, self.width(), self.img.width), np.linspace(self.height(), 0, self.img.height))
        return np.dstack((x,np.zeros(np.shape(x)),y))
        
    def earth_grid(self):
        """ give the position of each pixel in the earthframe """
        img_grid = self.image_grid()
        earth_grid = self.earth2flu(img_grid, True) + self.gps_pos
        return np.reshape(earth_grid, np.shape(img_grid))
        
    def predict_image(self, gps_pos, attitude):
        """ Give the prediction of an observation in a different position based on actual radar image """
        #TODO: 3D transformation of image (to take into account pitch and roll changes)
        exp_rot = rot.as_rotvec(self.attitude.inv()*attitude)[2]
        exp_rot_matrix = np.array([[np.cos(exp_rot), -np.sin(exp_rot)],[np.sin(exp_rot), np.cos(exp_rot)]])
        
        h = np.array([0, self.height()/2])
        exp_trans = self.earth2flu(gps_pos - self.gps_pos)[0:2]
        corr_trans = (exp_trans - h + exp_rot_matrix.dot(h))/self.precision
        
        warp_matrix = np.concatenate((exp_rot_matrix,np.array([[corr_trans[0]],[corr_trans[1]]])), axis = 1)
        predict_img = cv2.warpAffine(np.array(self.img), warp_matrix, self.img.size, flags=cv2.INTER_LINEAR, borderValue = 0);
        
        mask = cv2.warpAffine(np.array(self.img), warp_matrix, self.img.size, flags=cv2.INTER_LINEAR, borderValue = 255);
        diff = (mask - predict_img).astype(np.float16)
        diff[diff == 255] = np.nan
        prediction = diff + predict_img
                
        #TODO: just for test and vizualisation, could be removed()
        Image.fromarray(predict_img).save('radar2_2.png')
        
        return prediction
    
    def transformation_from(self, otherdata):
        """ Return the translation and the rotation based on the two radar images """
        #TODO: 3D transformation of image (to take into account pitch and roll changes)
        warp_mode = cv2.MOTION_EUCLIDEAN
        number_of_iterations = 5000;
        termination_eps = 1e-10;
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        (cc, warp_matrix) = cv2.findTransformECC (np.array(otherdata.img), np.array(self.img), warp_matrix, warp_mode, criteria)
        
        #TODO: just for test and vizualisation, could be removed()
        check_transform(self, warp_matrix, 'radar1_1.png')
        
        h1 = np.array([0, otherdata.height()/2])
        h2 = np.array([0, self.height()/2])
        
        img_trans = self.precision*np.array([warp_matrix[0,2], -warp_matrix[1,2]])
        abs_trans = h1 + img_trans + warp_matrix[0:2,0:2].T.dot(-h2)
        
        rotation = np.array([[warp_matrix[0,0], warp_matrix[0,1], 0], [warp_matrix[1,0], warp_matrix[1,1], 0], [0,0,1]])
        translation = -np.array([abs_trans[0], abs_trans[1], 0])
        return translation, rotation
    
    def image_position_from(self, otherdata):
        """ Return the actual position and attitude based on radar images comparison """
        translation, rotation = self.transformation_to(otherdata)
        
        gps_pos = otherdata.gps_pos + translation
        attitude = (self.attitude*rot.from_dcm(rotation)).as_quat()
        return gps_pos, attitude