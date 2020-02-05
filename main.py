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
reader = Reader('radardata2.h5', 69, 76)

# Creating Kalman filter for mapping
kalman = Kalman_Mapper()
#TODO: decide more precisely covariances
kalman.set_covariance(0.05, np.deg2rad(1), 0.04, np.deg2rad(1))
#kalman.set_covariance(0, 0, 900, 900)

# Creating recorder 
recorder = Recorder(reader, kalman)

for ts, radardata in reader:
    # add a new image
    pos, att = kalman.add(radardata)
    # save Kalman output
    recorder.record(ts)
    # update the map during mapping
    kalman.mapdata.show(pos)

# Extracting map after fusion
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
kalman = Kalman_Localizer("map_20200118_1620")
# Initialize the first position and attitude
kalman.init_position(reader.get_radardata(0).gps_pos, reader.get_radardata(0).attitude)

# Creating recorder 
recorder = Recorder(reader, kalman)

for ts, radardata in reader:
    # localize image (only radardata.img is used)
    pos, att = kalman.localize(radardata)
    print(np.linalg.norm(pos-radardata.gps_pos))
    # save Kalman output
    recorder.record(ts)
    # update the map during mapping
    #kalman.mapdata.show(pos)

# Plots
recorder.export_map()
'''