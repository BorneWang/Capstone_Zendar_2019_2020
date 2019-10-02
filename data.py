# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:36:34 2019

@author: Pierre-Louis
"""
from PIL import Image, ImageDraw
import numpy as np
from copy import deepcopy

def flu2earth(pos, q, inverse=False):
    """ Change of frame from front-left-up to earth frame """
    R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*q[1]*q[2]-2*q[0]*q[3], 2*q[1]*q[3]+2*q[0]*q[2]],
                  [2*q[1]*q[2]+2*q[0]*q[3], q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*q[2]*q[3]-2*q[0]*q[1]],
                  [2*q[1]*q[3]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
    if inverse:
        return R.dot(pos)
    else:
        return R.T.dot(pos)

def fur2flu(pos, inverse=False):
    """ Change of frame from front-up-right to earth front-left-up """
    R = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    if inverse:
        return R.dot(pos)
    else:
        return R.T.dot(pos)
    
def fur2earth(pos, q, inverse=False):
    """ Change of frame from front-up-right to earth frame """
    if inverse:
        return fur2flu(flu2earth(pos, q, inverse), inverse)
    else:
        return flu2earth(fur2flu(pos),q)
    
class Data:
    
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
        
    def image_grid(self):
        """ give the position of each pixel in the image frame """
        x, y = np.meshgrid(np.linspace(0, self.width(), self.img.width), np.linspace(self.height(), 0, self.img.height))
        return np.dstack((x,np.zeros(np.shape(x)),y))
        
    def earth_grid(self):
        """ give the position of each pixel in the earthframe """
        img_grid = self.image_grid()
        earth_grid = deepcopy(np.reshape(img_grid, (np.size(img_grid[:,:,0]),3)))
        for i in range(0,len(earth_grid)):
            earth_grid[i] = fur2earth(earth_grid[i], self.attitude)
        return np.reshape(earth_grid, np.shape(img_grid))
    
    def distance(self, other_data):
        """ Negative correlation between the two images, flattened to 1D """
        img_array1 = np.array(self.img)
        img_array2 = np.array(other_data.img)
        correl = np.corrcoef(img_array1.ravel(), img_array2.ravel())[0, 1]
        return -correl
        
    def circle(self):
        """ Take only data in the biggest centered circle that can fit in the image """
        alpha = Image.new('L', self.img.size,0)
        draw = ImageDraw.Draw(alpha)
        
        l = min(self.img.width,self.img.height)
        draw.pieslice([(self.img.width-l)/2,(self.img.height-l)/2, (self.img.width+l)/2, (self.img.height+l)/2],0,360,fill=255)
        img = Image.fromarray(np.logical_and(np.array(alpha.convert('1')), np.array(self.img)).astype('uint8')*255).convert('1')
        return Data(img, self.gps_pos, self.attitude)
    
    def meters2indices(self,x,y):
        """ Give the position of a pixel according its postopn in image frame """
        x_I = int(round(x/self.precision))
        y_I = int(round(y/self.precision))
        return x_I, y_I
    
    def intersection(self, other_data):
        """ Return the cropped data corresponding to the intersection of two datasets """
        
        # TODO: deal with case when the image does not have the same orientation
        m = fur2earth((self.gps_pos+other_data.gps_pos)/2, self.attitude, True)
        
        center1 = np.array([int(other_data.img.height/2), int(other_data.img.width/2)])
        m1 = center1 + other_data.meters2indices(m[0], m[2])
        center2 = np.array([int(self.img.height/2), int(self.img.width/2)])
        m2 = center2 - self.meters2indices(m[0], m[2])
        
        r = min(min(min(abs(m1-np.array([0, 0]))), min(abs(m1-np.array([other_data.img.width, other_data.img.height])))), min(min(abs(m2-np.array([0, 0]))), min(abs(m2-np.array([self.img.width, self.img.height])))))

        data_1 = self.crop(m2[0]-r, m2[1]+r, m2[0]+r, m2[1]-r)
        data_2 = other_data.crop(m1[0]-r, m1[1]+r, m1[0]+r, m1[1]-r)
        return data_1, data_2
    
    def crop(self, left, up, right, bottom):
        """ Return a crop of the actual data and its new absolute position and attitude """
        gps_pos = self.gps_pos + fur2earth(self.precision*np.array([bottom,0,left]), self.attitude)
        img = self.img.crop((left, self.img.height-up, right, self.img.height-bottom))
        return Data(img, gps_pos, self.attitude)
    
    