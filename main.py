import warnings
import numpy as np
from reader import Reader
from recorder import Recorder
from kalman import Kalman_Mapper_GPSCV2_3D, Kalman_Localizer

from utils import DBSCAN_filter

warnings.filterwarnings("ignore") 

# =============================================================================
# Preprocessing of images
# =============================================================================

def preprocessor(img):
    """ Succession of operation to apply to radar image for preprocessing """
    # comment and use radardat_ic if increasing contrast should be used
    img = DBSCAN_filter(img, kernel=(9,9), scale=1, binary=True) # use radardata_norm if used
    return img

# =============================================================================
# Mapping
# =============================================================================

# Loading data
reader = Reader('radardata2_norm.h5', 0, np.inf)

kalman = Kalman_Mapper_GPSCV2_3D(True) # Creating Kalman filter

#TODO: decide more precisely covariances
kalman.set_covariances(0.015, np.deg2rad(0.018), 0.035, np.deg2rad(0.099))

recorder = Recorder(reader, kalman) # Creating recorder 

for ts, radardata in reader:
    pos, att = kalman.add(radardata) # add a new image
    recorder.record(ts) # save Kalman output
    
    # update the displayed map during mapping
    if kalman.mapping:    
        kalman.mapdata.show(pos)

# Extracting map after fusion
if kalman.mapping:    
    m = kalman.mapdata 

# Plots
recorder.export_map()
recorder.plot_innovation()
recorder.plot_attitude()
recorder.plot_trajectory(False)
recorder.plot_kalman_evaluation()

# =============================================================================
# Localization
# =============================================================================
"""
# Loading data   
reader = Reader('radardata2.h5', 275, 360)

# Creating Kalman filter for mapping
kalman = Kalman_Localizer(False, "map_20200307_1744")
# Initialize the first position and attitude
# kalman.set_initial_position(reader.groundtruth['POSITION'][0], reader.groundtruth['ATTITUDE'][0])
kalman.set_initial_position(reader.get_radardata(0).gps_pos, reader.get_radardata(0).attitude)
# Creating recorder 
recorder = Recorder(reader, kalman)

for ts, radardata in reader:
    # localize image (only radardata.img is used)
    pos, att = kalman.localize(radardata)
    # save Kalman output
    recorder.record(ts)
    # see the map during localization
    #kalman.mapdata.show(pos)

# Plots
#recorder.export_map()
"""