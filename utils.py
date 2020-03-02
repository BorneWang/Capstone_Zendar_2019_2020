import cv2
import pyproj
import numpy as np
from PIL import Image
from pykml import parser
from copy import deepcopy
import scipy.stats as stat
from scipy.spatial.transform import Rotation as rot

def stat_test(Y, Yhat, S, p):
    """ Perform statistical test to reject outliers """
    used = np.zeros(len(Y))
    for i in range(0, len(Yhat)):
        if (Yhat[i] - Y[i])**2/S[i][i] <= stat.chi2.ppf(p, df=1):
            used[i] = Yhat[i] - Y[i]
    return used  

def stat_filter(x, p):
    """ Filter outliers from a sequence """
    mean = np.mean(x)
    std = np.std(x)
    out =  []
    for i in range(len(x)):
        if ((x[i] - mean)/std)**2 <= stat.chi2.ppf(p, df=1):
            out.append(x[i])
    return out

def import_kml(filename):
    """ Import KML file by retreiving timestamps and positions """
    out = []
    ts = []
    with open(filename) as f:
        root = parser.parse(f).getroot()
        pms = root.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
        for pm in pms:
            ts.append(float(pm.description.text.replace('\n',' ').split(' ')[1]))
            out.append( np.array(str(pm.findall('.//{http://www.opengis.net/kml/2.2}Point')[0].coordinates).split(',')).astype(np.double))
    for i in range(0,len(out)):
        out[i][0:2] = np.deg2rad(out[i][0:2])
    return ts, np.array(out)

def R(theta):
    """ Return 2D rotation matrix according rbd convention """
    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

def rotation_proj(attitude1, attitude2):
    """ Project the rotation only on Z axis from attitude1 to attitude2 """
    r = attitude1.apply(attitude2.apply([1,0,0],True))
    return rot.from_dcm(np.array([[r[0], -r[1], 0.],[r[1], r[0], 0.],[0., 0., 1.]]))

def ecef2lla(pos, inv=False):
    """ Convert position in ECEF frame to LLA """
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    if inv:
        if pos.ndim == 1:
            x, y, z = pyproj.transform(lla, ecef, pos[0], pos[1], pos[2], radians=True)
        else:        
            x, y, z = pyproj.transform(lla, ecef, pos[:,0], pos[:,1], pos[:,2], radians=True)
        return np.array([x, y, z]).T
    else:
        if pos.ndim == 1:
            lon, lat, alt = pyproj.transform(ecef, lla, pos[0], pos[1], pos[2], radians=True)
        else:        
            lon, lat, alt = pyproj.transform(ecef, lla, pos[:,0], pos[:,1], pos[:,2], radians=True)
        return np.array([lon, lat, alt]).T

def ecef2enu(lat0, lon0):
    """ Compute quaternion of transformation between LLA to ENU frame """
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
        return gps_positions - attitudes.apply(trans, True)
    else:          
        car_pos = []
        for i in range(len(gps_positions)):
            car_pos.append(gps_positions[i] - attitudes[i].apply(trans, True))
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
    im = np.power(sat*img/255, gamma)*255
    im[im >= 255] = 255
    return im