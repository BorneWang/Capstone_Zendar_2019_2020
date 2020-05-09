import warnings
import numpy as np
from copy import deepcopy
from reader import Reader
from recorder import Recorder
from preprocessor import Preprocessor
from kalman import Kalman_Mapper_CV2GPS_3D, Kalman_Localizer

from utils import rbd_translate

warnings.filterwarnings("ignore") 

# =============================================================================
# Preprocessing (must be done only once)
# =============================================================================
"""
src = '2019-11-20_1844_001_3VXs.h5'                     # source hdf5 file name
groundtruth_src = '2019-11-20_1844_001_3VXs_SBG.h5'     # groundtruth hdf5 file name

goal_name = 'radardata2_preprocessed.h5'                # name of output file

hdf5 = Preprocessor(src, goal_name, groundtruth_src)    # Initialize preprocessor
hdf5.run()                                              # Run preprocessor
"""
# =============================================================================
# Mapping
# =============================================================================

# User parameters
dataset_name = 'radardata2_preprocessed.h5'
start_time = 0
end_time = 265
build_map = False       # build map if true, if false calculate only positions
display_map = True      # when mapping (build_map = True), displays the map at the same time
use_groundtruth = True  # use groundtruth GPS value of SBG if true (because of problems in VN)
use_fusion = True       # if True use a Kalman filter fusion between CV2 and GPS mesurement (not used for now because of CV2 biases)
bias_estimation = True  # if True, perform an online estimation of CV2 biases

# Loading data
reader = Reader(dataset_name, start_time, end_time)

kalman = Kalman_Mapper_CV2GPS_3D(build_map, bias_estimation = bias_estimation) # Creating Kalman filter

recorder = Recorder(reader, kalman)         # Creating recorder 

for ts, radardata in reader:
    data = deepcopy(radardata)
    if use_groundtruth:
        data.gps_pos = reader.get_groundtruth_pos(ts)
        data.attitude = reader.get_groundtruth_att(ts)
               
    pos, att = kalman.add(data, use_fusion) # add a new image to the map
        
    recorder.record(ts)                     # save Kalman output    
    if kalman.mapping and display_map:      
        kalman.mapdata.show(rbd_translate(pos, att, reader.tracklog_translation))            # update the displayed map during mapping

# Extracting map after fusion
if kalman.mapping:    
    m = kalman.mapdata 

# Plots
#recorder.export_map(gps_only = not use_fusion)
recorder.plot_trajectory(arrow = False , cv2 = use_fusion)
recorder.plot_attitude(cv2 = use_fusion)
recorder.plot_altitude(cv2 = use_fusion)
if use_fusion:
    recorder.plot_innovation()
    recorder.plot_kalman_evaluation(use_groundtruth)

# =============================================================================
# Localization
# =============================================================================
"""
# User parameters
dataset_name = 'radardata2_preprocessed.h5'
start_time = 265
end_time = 360
map_to_use = "radardata2_firstpass"     # name of the map to use (should be in a folder maps/)
mapping = False                         # update the map at the same as localizing (SLAM)
display_map = True                     # when mapping, displays the map at the same time
use_groundtruth = True                  # use groundtruth GPS value of SBG if true (because of problems in VN)
gps_guess = False                       # use always the gps as a first guess for localization


# Loading data   
reader = Reader(dataset_name, start_time, end_time)

kalman = Kalman_Localizer(mapping, map_to_use)          # Creating Kalman filter for mapping
# Initialize the first position and attitude
kalman.set_initial_position(reader.get_groundtruth_pos(0), reader.get_groundtruth_att(0))

recorder = Recorder(reader, kalman)                     # Creating recorder 

for ts, radardata in reader:
    data = deepcopy(radardata)
    if use_groundtruth:
        data.gps_pos = reader.get_groundtruth_pos(ts)
        data.attitude = reader.get_groundtruth_att(ts)
    pos, att = kalman.localize(data, gps_guess)         # localize image 
    
    recorder.record(ts)                      # save Kalman output
    if display_map:
        data = deepcopy(radardata)
        data.gps_pos, data.attitude = pos, att
        kalman.mapdata.show(rbd_translate(pos, att, reader.tracklog_translation), data)             # see the map during localization

# Plots
#recorder.export_map()
#recorder.plot_attitude()
#recorder.plot_trajectory(False)
recorder.plot_kalman_evaluation()
"""