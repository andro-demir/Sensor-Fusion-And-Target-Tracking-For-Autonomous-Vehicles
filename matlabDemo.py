# matlabDemo.py
import numpy as np
import argparse
from objectClasses import Obstacle, ObjectListCls
import objectAssociation as assc
from helper_functions import kf_measurement_update, temporal_alignment


def main(time, Measurements, States, last_update_times):
    '''
    param: time (float)
    param: Measurements (2d array) -- Sensor Obstacle List at t
    param: States (2d array) -- Fusion List at t-1
    return: stateEstimates (2d array)-- Fusion List at t
    Measurements in Matlab corresponds to sensorObjList in Python and
    States in Matlab corresponds to fusionList in Python and
    '''
    # We created the fusionList at time,
    # Get the sensorObjectList at time+1
    # Note: In Eatron's code Measurements = [pos_x, v_x, pos_y, v_y]'
    args = parse_args()
    sensor_specs={
        'pos_initializers': np.array((100., 0, 0)),
        'vel_initializers': np.array((0., 0, 0))}
    sensorObjList = ObjectListCls(time, sensor_specs)
    measurementNoise = np.zeros((11, 11))
    measurementNoise[:] = np.nan
    measurementNoise[:6,:6] = np.array([[22.1, 0, 0, 0, 0, 0], [0, 22.1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0], [0, 0, 0, 2209, 0, 0],
                                        [0, 0, 0, 0, 2209, 0], [0, 0, 0, 0, 0, 1]])

    for measurement in Measurements.T:
        sensorObjList.append(Obstacle(pos_x=measurement[0],
                                      pos_y=measurement[2],
                                      pos_z=None, v_x=measurement[1],
                                      v_y=measurement[3], v_z=None,
                                      a_x=None, a_y=None, a_z=None,
                                      yaw=None, r_yaw=None, P=measurementNoise))
    fusionList = ObjectListCls(time)
    for idx, state in enumerate(States.T):
        fusionList.append(Obstacle(pos_x=state[0], pos_y=state[2],
                                   pos_z=None, v_x=state[1],
                                   v_y=state[3], v_z=None,
                                   a_x=None, a_y=None, a_z=None,
                                   yaw=None, r_yaw=None, P=measurementNoise,
                                   last_update_time=last_update_times[idx]))

    mahalanobisMatrix = assc.getMahalanobisMatrix(fusionList, sensorObjList)
    rowInd, colInd, cluttered_matches, cleaned_matches, \
                                    num_true_positive = assc.matchObjs(
                                                         mahalanobisMatrix, 
                                                       args.clutter_threshold)
    kf_measurement_update(fusionList, sensorObjList, (rowInd, colInd))
    # Probability of existence of obstacles is updated:
    fusionList = assc.updateExistenceProbability(fusionList,
                                                 sensorObjList,
                                                 rowInd, colInd,
                                                 cluttered_matches, 
                                                 args.last_seen,
                                                 args.distance_to_ego)
    N_obstacles = len(fusionList)
    print("Number of tracked actors:", N_obstacles)
    stateEstimates = np.zeros((4, N_obstacles))  # (pos_x, vel_x, pos_y, vel_y)
    last_update_times = np.zeros((1, N_obstacles))
    for i in range(N_obstacles):
        stateEstimates[0, i] = fusionList[i].s_vector[0]  # pos_x
        stateEstimates[1, i] = fusionList[i].s_vector[3]  # v_x
        stateEstimates[2, i] = fusionList[i].s_vector[1]  # pos_y
        stateEstimates[3, i] = fusionList[i].s_vector[4]  # v_y
        last_update_times[0, i] = fusionList[i].last_update_time

    print(50 * "**")
    print("Time: %f" % time)
    print("Measurements:\n", Measurements)
    print("State Estimates:\n", stateEstimates)
    print("Last Update Times:\n", last_update_times)
    
    trackedStates = [x[1] for x in cleaned_matches]
    return [stateEstimates, last_update_times, trackedStates, 
            num_true_positive]

def parse_args():
    parser = argparse.ArgumentParser(description='CSL-EATRON KF TRACKER.', 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--clutter_threshold', type=float, default=0.75, 
                        help='if mahalanobis distance > clutter thresholod,'
                             'assign as false positive')
    parser.add_argument('--last_seen', type=float, default=1.0, 
                        help='if the tracked object has not been seen longer'
                             'than last_seen, delete it from the fusion list')
    parser.add_argument('--distance_to_ego', type=float, default=200, 
                        help='distance to ego (L1 norm of the tracked objects'
                             'state vector)')
    args = parser.parse_args()
    return args


