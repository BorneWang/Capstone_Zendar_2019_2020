import warnings
import numpy as np
from copy import deepcopy
from reader import Reader
from recorder import Recorder
from kalman import Kalman_Mapper_GPSCV2_3D, Kalman_Localizer

warnings.filterwarnings("ignore") 

# =============================================================================
# Mapping
# =============================================================================

# User parameters
start_time = 10
end_time = 75
build_map = False        # build map if true, if false calculate only positions
use_groundtruth = True  # use groundtruth GPS value of VN if true (because of problems in VN)
use_fusion = False      # if True use a Kalman filter fusion between CV2 and GPS mesurement (not used for now because of CV2 biases)
display_map = True      # when mapping, displays the map at the same time

# Loading data
reader = Reader('radardata2_preprocessed.h5', start_time, end_time)

kalman = Kalman_Mapper_GPSCV2_3D(build_map) # Creating Kalman filter
# Setting covariance based on GPS and CV2 performance analysis (could be tuned)
gps_pos_std = 0.015
gps_att_std = np.deg2rad(0.018)
cv2_pos_std = 0.035
cv2_att_std = np.deg2rad(0.099)
kalman.set_covariances(gps_pos_std, gps_att_std, cv2_pos_std, cv2_att_std)

recorder = Recorder(reader, kalman) # Creating recorder 

for ts, radardata in reader:
    data = deepcopy(radardata)
    if use_groundtruth:
        data.gps_pos = reader.get_groundtruth_pos(ts)
        data.attitude = reader.get_groundtruth_att(ts)
               
    pos, att = kalman.add(data, use_fusion) # add a new image
        
    recorder.record(ts) # save Kalman output    
    # update the displayed map during mapping
    if kalman.mapping and display_map:    
        kalman.mapdata.show(pos)

# Extracting map after fusion
if kalman.mapping:    
    m = kalman.mapdata 

# Plots
recorder.export_map()
recorder.plot_attitude()
recorder.plot_trajectory(False)
if use_fusion:
    recorder.plot_innovation()
    recorder.plot_kalman_evaluation()

# =============================================================================
# Localization
# =============================================================================
"""
# User parameters
start_time = 275
end_time = 360
map_to_use = "map_20200307_1744"    # name of the map to use (should be in a folder maps/)
mapping = False                     # update the map at he same as localizing (SLAM)
display_map = True                  # when mapping, displays the map at the same time


# Loading data   
reader = Reader('radardata2_preprocessed.h5', start_time, end_time)

# Creating Kalman filter for mapping
kalman = Kalman_Localizer(mapping, map_to_use)
# Initialize the first position and attitude
kalman.set_initial_position(reader.groundtruth['POSITION'][0], reader.groundtruth['ATTITUDE'][0])
# Creating recorder 
recorder = Recorder(reader, kalman)

for ts, radardata in reader:
    # localize image (only radardata.img is used)
    pos, att = kalman.localize(radardata)
    # save Kalman output
    recorder.record(ts)
    # see the map during localization
    if display_map:
        kalman.mapdata.show(pos)

# Plots
recorder.export_map()
recorder.plot_attitude()
recorder.plot_trajectory(False)
"""