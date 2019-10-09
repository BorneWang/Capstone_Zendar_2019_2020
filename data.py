from PIL import Image, ImageDraw
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot

def earth2flu(pos, q, inverse=False):
    """ Change of frame from earth frame to front-left-up """
    quaternion = rot.from_quat(q)
    return quaternion.apply(pos, inverse)

class RadarData:
    
    def __init__(self, img, gps_pos, attitude, precision=0.04):
        self.img = img
        self.precision = precision
        self.gps_pos = gps_pos
        self.attitude = attitude
        
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
    
    def image_grid(self):
        """ give the position of each pixel in the image frame """
        x, y = np.meshgrid(np.linspace(0, self.width(), self.img.width), np.linspace(self.height(), 0, self.img.height))
        return np.dstack((x,np.zeros(np.shape(x)),y))
        
    def earth_grid(self):
        """ give the position of each pixel in the earthframe """
        img_grid = self.image_grid()
        earth_grid = earth2flu(img_grid, self.attitude, True) + self.gps_pos
        return np.reshape(earth_grid, np.shape(img_grid))
        
    def circle(self):
        """ Take only data in the biggest centered circle that can fit in the image """
        alpha = Image.new('L', self.img.size,0)
        draw = ImageDraw.Draw(alpha)
        
        l = min(self.img.width,self.img.height)
        draw.pieslice([(self.img.width-l)/2,(self.img.height-l)/2, (self.img.width+l)/2, (self.img.height+l)/2],0,360,fill=255)
        img = Image.fromarray(np.logical_and(np.array(alpha.convert('1')), np.array(self.img)).astype('uint8')*255).convert('1')
        return RadarData(img, self.gps_pos, self.attitude)
    
    def distance(self, other_data):
        """ Negative correlation between the two images, flattened to 1D """
        img_array1 = np.array(self.img)
        img_array2 = np.array(other_data.img)
        correl = np.corrcoef(img_array1.ravel(), img_array2.ravel())[0, 1]
        return -correl
    
    def projection(self, other_data):       
        """ Project an image on an other image plane (x,y) """
        rot_vec = (rot.from_quat(self.attitude).inv()*rot.from_quat(other_data.attitude)).as_rotvec()
        proj_q = rot.from_rotvec(np.array([rot_vec[0], rot_vec[1], 0]))
        R = proj_q.as_dcm()[0:2,0:2]
        if np.array_equal(R, np.eye(2)):
            return deepcopy(self)
        else:
            ru_point = R.dot(np.array([self.height(), self.width()]))
            img = self.img.transform(other_data.meters2indices(ru_point), Image.AFFINE, [R[0,0], R[0,1], 0, R[1,0], R[1,1], 0])
            return RadarData(img, self.gps_pos, rot.from_rotvec(np.array([0, 0, rot_vec[2]])).as_quat())
    
    def intersection(self, other_data):
        """ Return the cropped data corresponding to the intersection of two datasets """ 
        proj_data = self.projection(other_data)
        
        c = earth2flu(earth2flu(np.array([proj_data.width()/2, proj_data.height()/2, 0]), proj_data.attitude, True) + proj_data.gps_pos, other_data.attitude)
        
        m = (c + np.array([other_data.width()/2, other_data.height()/2, 0]))/2
        m1 = other_data.meters2indices(np.array([m[0], m[1]]))

        m = earth2flu(earth2flu(m, other_data.attitude, True) - proj_data.gps_pos, proj_data.attitude)
        m2 = proj_data.meters2indices(np.array([m[0], m[1]]))
        
        r = min(min(min(m1), min(abs(m1-np.array([other_data.img.width, other_data.img.height])))), min(min(m2), min(abs(m2-np.array([proj_data.img.width, proj_data.img.height])))))
        
        data_1 = proj_data.crop(m2[0]-r, m2[1]+r, m2[0]+r, m2[1]-r)
        data_2 = other_data.crop(m1[0]-r, m1[1]+r, m1[0]+r, m1[1]-r)
        return data_1, data_2
    
    def crop(self, left, up, right, bottom):
        """ Return a crop of the actual data and its new absolute position and attitude """
        gps_pos = self.gps_pos + earth2flu(self.precision*np.array([bottom,left,0]), self.attitude, True)
        img = self.img.crop((left, self.img.height-up, right, self.img.height-bottom))
        return RadarData(img, gps_pos, self.attitude)
    
    