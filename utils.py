import cv2
import pyproj
import numpy as np
from PIL import Image
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot

def rotation_proj(attitude1, attitude2):
    """ Project the rotation only on Z axis """
    r = attitude1.apply(attitude2.apply([0,1,0],True))
    r[r>=1] = 1
    theta = np.mean([np.arcsin(abs(r[0])), np.arccos(abs(r[1]))])
    theta = np.sign(np.arcsin(abs(r[0])))*theta
    sin = np.sin(theta)
    cos = np.cos(theta)
    return rot.from_dcm(np.array([[ cos , -sin , 0. ],
                     [ sin , cos  , 0. ],
                     [  0.  ,  0.   , 1. ]]))

def ecef2lla(pos):
    """ Convert position in ECEF frame to LLA """
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    if pos.ndim == 1:
        lon, lat, alt = pyproj.transform(ecef, lla, pos[0], pos[1], pos[2], radians=True)
    else:        
        lon, lat, alt = pyproj.transform(ecef, lla, pos[:,0], pos[:,1], pos[:,2], radians=True)
    return np.array([lon, lat, alt]).T

def ecef2enu(lat0, lon0):
    """ Compute quaternion of transformation between ECEF to ENU frame """
    MatNorthPole = np.array([[-1., 0., 0.],
                           [0., -1., 0.],
                           [ 0., 0. , 1.]])
    sColat = np.sin(np.pi/2-lat0)
    cColat = np.cos(np.pi/2-lat0)
    MatLat = np.array([[ cColat , 0. , sColat ],
                     [   0.   , 1. ,   0.   ],
                     [-sColat , 0. , cColat ]])
    sLon = np.sin(lon0)
    cLon = np.cos(lon0)
    Matlon = np.array([[ cLon , -sLon , 0. ],
                     [ sLon , cLon  , 0. ],
                     [  0.  ,  0.   , 1. ]])
    return rot.from_dcm(np.array([[0,-1,0],[1,0,0],[0,0,1]]).dot(Matlon.dot(MatLat.dot(MatNorthPole)).T))

def rbd_translate(gps_positions, attitudes, trans):
    """ Convert the position of the top left corner of image to car position """
    if gps_positions.ndim == 1:
        return attitudes.apply(attitudes.apply(gps_positions) - trans, True)
    else:          
        car_pos = []
        for i in range(len(gps_positions)):
            car_pos.append(attitudes[i].apply(attitudes[i].apply(gps_positions[i]) - trans, True))
        return np.array(car_pos)
    
def check_transform(data, rotation, translation, name):
    """ Save an image to vizualize the calculated transformation (for test purpose) """
    translation = translation/0.04
    shape = (np.shape(data.img)[1], np.shape(data.img)[0])
    warp_matrix = np.concatenate(((rotation.as_dcm()[:2,:2]).T,np.array([[-translation[0]],[-translation[1]]])), axis = 1)
    Image.fromarray(cv2.warpAffine(data.img, warp_matrix, shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)).save(name);    

def merge_img(img1, img2, P1, P2, P_start, P_end):
    """ Merge two images pixel by pixel, weighted by uncertainty, only in modified area """
    img = deepcopy(img1)
    cov_img = deepcopy(P1)
    for i in range(max(0,P_start[1]), min(P_end[1], np.size(img1, 0))):
        for j in range(max(0,P_start[0]), min(P_end[0], np.size(img1, 1))):
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

def increase_saturation(img):
    """ Increase saturation of an image """
    sat = 1.7
    gamma = 1.2
    im = np.power(img/(255/sat), gamma)*255
    im[im >= 255] = 255
    return im

def image_overlap(data1,data2):
    
    w1 = np.ones(np.shape(data1.img))
    w2 = np.ones(np.shape(data2.img))
    
    white_1 = RadarData(0,w1,data1.gps_pos,data1.attitude)
    white_2 = RadarData(0,w2,data2.gps_pos,data2.attitude)
    
    mask1 = white_1.predictimage(data2.gps_pos,data2.attitude)
    mask2 = white_2.predictimage(data1.gps_pos,data1.attitude)
    
    out1 = np.ones(np.shape(data2.img))
    out1[mask1==1] = data2.img
    
    out2 = np.ones(np.shape(data1.img))
    out2[mask2==1] = data1.img
    


