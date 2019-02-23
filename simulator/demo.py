import numpy as np
from objectClasses.objectClasses import SimSensor

# create environment sensors using Matlab file.
cam_rear = SimSensor('./cam_rear.mat')
cam_front = SimSensor('./cam_front.mat')
radar_rear = SimSensor('./radar_rear.mat')
radar_front = SimSensor('./radar_front.mat')

time_frame = [cam_rear.list_time, cam_front.list_time,
              radar_rear.list_time, radar_rear.list_time]

# Create the time frame from unique entities of the times recorded for each
# sensor. This will serve as the time scale for the rest of the system.
time_frame = list(np.unique(np.concatenate(time_frame)))

for time in time_frame:

    list_object_cam_rear, _ = cam_rear.return_obstacle_list(time)
    list_object_cam_front, _ = cam_front.return_obstacle_list(time)
    list_object_radar_front, _ = radar_rear.return_obstacle_list(time)
    list_object_radar_rear, _ = radar_front.return_obstacle_list(time)

