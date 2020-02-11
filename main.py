import warnings
import numpy as np
from reader import Reader
from recorder import Recorder
from kalman import Kalman_Mapper, Kalman_Localizer

warnings.filterwarnings("ignore") 

# =============================================================================
# Mapping
# =============================================================================

# Loading data   
reader = Reader('radardata2.h5', 10, 75)


kalman = Kalman_Mapper(False) # Creating Kalman filter for mapping

#TODO: decide more precisely covariances
kalman.set_covariance(0.05, np.deg2rad(1), 0.04, np.deg2rad(1))

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
#recorder.plot_innovation()
#recorder.plot_attitude()

# =============================================================================
# Localization
# =============================================================================
'''
# Loading data   
reader = Reader('radardata2.h5', 6, 11)

# Creating Kalman filter for mapping
kalman = Kalman_Localizer(False, "map_20200210_1635")
# Initialize the first position and attitude
kalman.init_position(reader.get_radardata(0).gps_pos, reader.get_radardata(0).attitude)

# Creating recorder 
recorder = Recorder(reader, kalman)

for ts, radardata in reader:
    # localize image (only radardata.img is used)
    pos, att = kalman.localize(radardata)
    # save Kalman output
    recorder.record(ts)
    # update the map during mapping
    #kalman.mapdata.show(pos)

# Plots
recorder.export_map()
'''