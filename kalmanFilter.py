import numpy as np
import sys

def kf_measurement_update(fusion_obj, sensor_obj):
    """
    When there is a new measurement in the sensor, run the following kalman
    filter equations to update the fusion object's state and covariance. 
    Make sure to run temporal_alignment() on the fusion_obj 
    before calling this function.

    :param fusion_obj: (class) the object being tracked (in the fusion) with 
    the properties: x (current state), P (state covariance matrix)
    :param sensor_obj: (class) the object being tracked (in the sensor) with 
    the properties: x (current state), H (the observation model), 
    R (measurement noise covariance matrix)
    :return:
    """
    # throw and error message if time is not aligned.
    if fusion_obj.timeStamp != sensor_obj.timeStamp: 
        sys.exit("temporal_alignment() should be run on fusion_obj before"
                 "calling kalman_measurement_update (fusion_obj and "
                 "sensor_obj should have the same timeStamps)")

    # kalman filter equations:
    # R is the cov of the obs noise
    S = np.dot(np.dot(sensor_obj.H, fusion_obj.P), sensor_obj.H.T) + \      
                                                      sensor_obj.R  
    # K is the kalman gain
    K = np.dot(np.dot(fusion_obj.P, sensor_obj.H.T), np.linalg.inv(S))  
    # Updated aposteriori state estimate
    x = fusion_obj.x + np.dot(K,sensor_obj.x - np.dot(sensor_obj.H,
                                                      fusion_obj.x))  
    # Updated aposteriori estimate covariance
    P = np.dot(np.eye(fusion_obj.x.shape[0]) - np.dot(K, sensor_obj.H), 
                                                         fusion_obj.P)  
    # update global object state and covariance
    fusion_obj.P = P
    fusion_obj.x = x
    pass

debug = False

if debug:
    from temporalAlignment import *
    class object():  # dummy object class
        def __init__(self, is_sensor=False):
            self.timeStamp = 0
            self.x = np.random.random((8,))  # random state
            self.P = np.random.random((8,8))  # random cov
            delta = 1  # !
            self.F = np.array([[1, 0, delta, 0, 0.5*delta**2, 0, 0, 0],
                              [0, 1, 0, delta, 0, 0.5*delta**2, 0, 0],
                              [0, 0, 1, 0, delta, 0, 0, 0],
                              [0, 0, 0, 1, 0, delta, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, delta],
                              [0, 0, 0, 0, 0, 0, 0, 0]])
            # zeros for constant vel model, input should also change:
            self.u = np.zeros((8,))  
            self.w = np.random.random((8,))  # process noise
            Q = np.zeros((8,8))
            # noise added only at the last derivatives:
            Q[3:5, 3:5] = np.random.random() * np.eye(2)  
            Q[7, 7] = np.random.random()
            self.Q = Q

            # add the other params for sensor: this is a dummy example sensor 
            # sensor and fusion object might be different
            if is_sensor:  
                self.R = np.random.random((8,8))
                self.H = np.random.random((8,8))
                self.timeStamp = 3  # assuming this is the new measurement time
            pass

    fusion_obj = object()
    sensor_obj = object(is_sensor=True)

    # try updating fusion obj without time alignment:
    try:
        kf_measurement_update(fusion_obj, sensor_obj)
    except:
        print('Error caught because of not calling the temporal alignment')

        print('\nFusion object state before update: ', fusion_obj.x)
        temporal_alignment(fusion_obj, sensor_obj.timeStamp)
        kf_measurement_update(fusion_obj, sensor_obj)
        print('Fusion object state after update: ', fusion_obj.x)



