import numpy as np
from objectClasses import Obstacle

def spatial_alignment(obj_list, H_sensor_veh):
    """
    Transform the state vector of the object from sensor coordinate frame
    to vehicle coordinate frame. Update state and cov of the objects.
    :param obj_list: (list) a list that contains obstacles(class),
    :param H_sensor_veh: Transformation matrix from sensor to vehicle coordinate frame
    :return:
    """
    for obj in obj_list:
        obj.s_vector = np.dot(H_sensor_veh, np.concatenate((obj.s_vector, [1])))[:-1]
        obj.P = np.dot(np.dot(H_sensor_veh[:-1, :-1], obj.P),
                       H_sensor_veh[:-1, :-1].T)
    pass


def temporal_alignment(obj_list, current_time, method='SingleStep'):
    """
    preliminary kalman filter update for the obj.
    :param obj_list: (class) a list that contains obstacles(class), with property:
                    timeStamp (this is the time of the objects in the list)
    :param current_time:
    :param method: (str) Method for integral, if 'SingleStep' single integral
                   will be taken with delta equal to time
                   difference, if 'EqualStep' for every unit between
                   current time and object time one integral will be taken.
    :return:
    """

    def alignment_equations(obj, delta=1.):
        if obj.s_vector.shape[0] == 8:  # z axis is not included
            F = np.array([[1, 0, delta, 0, 0.5 * delta ** 2, 0, 0, 0],
                          [0, 1, 0, delta, 0, 0.5 * delta ** 2, 0, 0],
                          [0, 0, 1, 0, delta, 0, 0, 0],
                          [0, 0, 0, 1, 0, delta, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, delta],
                          [0, 0, 0, 0, 0, 0, 0, 0]])

            w = np.zeros((8,))
            w[4:6] = np.random.normal(size=(2,))  # noise added to accelerations
            accs = range(4, 6)
            Q = np.zeros((8, 8))
            Q[accs, accs] = np.multiply(np.random.normal(size=(2, 2)), 
                                        np.eye(2))  
            # noise added only at the last derivatives:
            Q[-1, -1] = np.random.normal()
        else:  # z axis is included
            F = np.array([[1, 0, 0, delta, 0, 0, 0.5 * delta ** 2, 0, 0, 0, 0],
                          [0, 1, 0, 0, delta, 0, 0, 0.5 * delta ** 2, 0, 0, 0],
                          [0, 0, 1, 0, 0, delta, 0, 0, 0.5 * delta ** 2, 0, 0],
                          [0, 0, 0, 1, 0, 0, delta, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, delta, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, delta, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, delta],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            w = np.zeros((11,))
            accs = range(6, 9)
            w[accs] = np.random.normal(size=(3,))  # noise added to accelerations

            Q = np.zeros((11, 11))
            Q[6:9, 6:9] = np.multiply(np.random.normal(size=(3, 3)), np.eye(
                3))  # noise added only at the last derivatives:
            Q[-1, -1] = np.random.normal()

        not_nan_idx = np.where(np.invert(np.isnan(obj.s_vector)))[0]

        s_vector_valid = obj.s_vector[not_nan_idx]
        P_valid = obj.P[not_nan_idx, :][:, not_nan_idx]
        F_valid = F[not_nan_idx, :][:, not_nan_idx]
        Q_valid = Q[not_nan_idx, :][:, not_nan_idx]

        obj.s_vector[not_nan_idx] = np.dot(F_valid, s_vector_valid) + obj.u[
            not_nan_idx] + w[not_nan_idx]

        obj.P[not_nan_idx, :][:, not_nan_idx] = np.dot(np.dot(F_valid, P_valid),
                                                       F_valid.T) + Q_valid

    if method == 'SingleStep':
        for obj in obj_list:
            delta = current_time - obj_list.timeStamp  # update delta
            alignment_equations(obj, delta=delta)
        obj_list.timeStamp = current_time

    elif method == 'EqualStep':
        delta = 1  # ! TODO: is this a correct precision??
        for obj in obj_list:
            for _ in range(obj_list.timeStamp + delta, current_time + delta, delta):
                alignment_equations(obj, delta=delta)
        obj_list.timeStamp = current_time

    pass


def kf_measurement_update(fusion_obj_list, sensor_obj_list, association_indices):
    """
    When there is a new measurement in the sensor, run the following kalman
    filter equations to update the fusion object's state and covariance.
    Make sure to run temporal_alignment() on the fusion_obj
    before calling this function.

    :param fusion_obj_list: (class) the object being tracked (in the fusion) with
    the properties: s_vector (current state), P (state covariance matrix)
    :param sensor_obj_list: (list) the object being tracked (in the sensor) with
    the properties: s_vector (current state), H (the observation model),
    P (measurement noise covariance matrix)
    :param association_indices: (list,list), indicies of object association in the lists
    :return:
    """
    column = association_indices[0]
    row = association_indices[1]
    for idx in range(len(column)):
        # TODO: check with Andac if the idx are correct
        sensor_obj = sensor_obj_list[column[idx]]
        fusion_obj = fusion_obj_list[row[idx]]

        # remove the rows and columns with nans
        s_vector_s = np.copy(sensor_obj.s_vector)
        s_vector_f = np.copy(fusion_obj.s_vector)
        P_f = np.copy(fusion_obj.P)
        P_s = np.copy(sensor_obj.P)
        # not_nan_idx = np.where(np.invert(np.isnan(s_vector_s)))[0]

        not_nan_idx_fus = set(np.where(np.invert(np.isnan(s_vector_f)))[0])
        not_nan_idx_sens = set(np.where(np.invert(np.isnan(s_vector_s)))[0])

        if s_vector_s.shape[0] == 8:
            miss_accs_fusion = {i for i in range(4, 6) if not i in not_nan_idx_fus}
            miss_accs_sensor = {i for i in range(4, 6) if not i in not_nan_idx_sens}
        else:
            miss_accs_fusion = {i for i in range(6, 9) if not i in not_nan_idx_fus}
            miss_accs_sensor = {i for i in range(6, 9) if not i in not_nan_idx_sens}

        # for the missing accelerations, initialize with random
        # numbers but give huge uncertanity for them
        if miss_accs_fusion:  
            s_vector_f[list(miss_accs_fusion)] = 0.001 * np.random.normal(
                size=len(miss_accs_fusion))
            maxs = max(list(miss_accs_fusion))
            mins = min(list(miss_accs_fusion))
            P_f[mins:maxs + 1, :maxs + 1] = 0.
            P_f[:maxs + 1, mins:maxs + 1] = 0.
            P_f[mins:maxs + 1, mins:maxs + 1] = 1e18 * np.eye(maxs - mins + 1)
        
        # if sensor is missing accs, use the last estimate of the
        # state as measuremnt with huge uncertanity around it
        if miss_accs_sensor:  
            maxs = max(list(miss_accs_sensor))
            mins = min(list(miss_accs_sensor))
            P_s[mins:maxs + 1, :maxs + 1] = 0.
            P_s[:maxs + 1, mins:maxs + 1] = 0.
            P_s[mins:maxs + 1, mins:maxs + 1] = 1e18 * np.eye(maxs - mins + 1)
            if miss_accs_fusion:
                s_vector_s[list(miss_accs_sensor)] = 0.001 * np.random.normal(
                    size=len(miss_accs_sensor))
            else:
                s_vector_s[list(miss_accs_sensor)] = s_vector_f[list(miss_accs_sensor)]

        not_nan_idx_fus = set(np.where(np.invert(np.isnan(s_vector_f)))[0])
        not_nan_idx_sens = set(np.where(np.invert(np.isnan(s_vector_s)))[0])

        not_nan_idx = np.array(
            list(not_nan_idx_fus.intersection(not_nan_idx_sens)))

        # get the idx of first measurements if there is any
        first_time_measurement_idx = np.array(list(not_nan_idx_sens - not_nan_idx_fus))

        s_vector_s = s_vector_s[not_nan_idx]
        s_vector_f = s_vector_f[not_nan_idx]

        P_f = np.copy(P_f)[not_nan_idx, :][:, not_nan_idx]
        P_s = np.copy(P_s)[not_nan_idx, :][:, not_nan_idx]
        H = sensor_obj.H[not_nan_idx, :][:, not_nan_idx]

        # kalman filter equations:
        # P_s is the cov of the obs noise
        S = np.dot(np.dot(H, P_f), H.T) + P_s
        # K is the kalman gain
        K = np.dot(np.dot(P_f, H.T), np.linalg.inv(S))
        # Updated aposteriori state estimate
        x = s_vector_f + np.dot(K, s_vector_s - np.dot(H, s_vector_f))
        # Updated aposteriori estimate covariance
        P = np.dot(np.eye(s_vector_f.shape[0]) - np.dot(K, H), P_f)

        # update global object state and covariance
        fusion_obj.P[not_nan_idx[:, np.newaxis], not_nan_idx] = P

        fusion_obj.s_vector[not_nan_idx] = x

        # update the last update time of the fusion object
        fusion_obj.last_update_time = sensor_obj_list.timeStamp

        # if there is any new measurements write them to fusion state
        if len(first_time_measurement_idx):
            fusion_obj.s_vector[first_time_measurement_idx] = sensor_obj.s_vector[
                first_time_measurement_idx]
            fusion_obj.P[
                first_time_measurement_idx[:, np.newaxis], first_time_measurement_idx] = \
                sensor_obj.P[
                    first_time_measurement_idx[:,
                    np.newaxis], first_time_measurement_idx]
    pass


def initialize_fusion_objects(not_assigned_sensor_obj_list):
    """
    :param not_assigned_sensor_obj_list: list of not assigned objects from the sensors
    :return: fusion_list_initialized_objects:
    """
    sensor_specs = not_assigned_sensor_obj_list.sensor_specs
    time = not_assigned_sensor_obj_list.timeStamp
    new_fusion_elements = []
    
    for sensor_obj in not_assigned_sensor_obj_list:
        s_vector = sensor_obj.s_vector
        P = sensor_obj.P
        
        if not all(s_vector[:3] == s_vector[:3]):  # some missing position measurements
            pos_initializers = sensor_specs['pos_initializers']
            pos_nans = np.where(np.isnan(s_vector[:3]))[0]
            for i in pos_nans:
                s_vector[:3][i] = np.random.normal(loc=pos_initializers[i],
                                                   scale=pos_initializers[i] / 10.)
                P[i, :] = 0.
                P[:, i] = 0.
                P[i, i] = 1e18
        
        if not all(s_vector[3:6] == s_vector[3:6]):  # some missing velocity measurements
            vel_initializers = sensor_specs['vel_initializers']
            vel_nans = np.where(np.isnan(s_vector[3:6]))[0]
            for i in vel_nans:
                s_vector[3:6][i] = np.random.normal(loc=vel_initializers[i],
                                                    scale=vel_initializers[i] / 10.)
                P[3+i, :] = 0.
                P[:, 3+i] = 0.
                P[3+i, 3+i] = 1e18

        if not all(s_vector[6:9] == s_vector[6:9]):  # some missing acc measurements
            acc_nans = np.where(np.isnan(s_vector[6:9]))[0]
            s_vector[6:9][acc_nans] = np.random.normal(size=(len(acc_nans)))
            for i in acc_nans:
                P[6+i, :] = 0.
                P[:, 6+i] = 0.
                P[6+i, 6+i] = 1e18
        new_fusion_elements.append(Obstacle(pos_x=s_vector[0], pos_y=s_vector[1],
                                            pos_z=s_vector[2], v_x=s_vector[3],
                                            v_y=s_vector[4], v_z=s_vector[5],
                                            a_x=s_vector[6], a_y=s_vector[7],
                                            a_z=s_vector[8],
                                            yaw=s_vector[9], r_yaw=s_vector[10], P=P,
                                            last_update_time=time, p_existence=1))
    return new_fusion_elements


def drop_objects(fusion_list, distance_to_ego=100):
    """
    :param fusion_list:
    :distance_to_ego: 
    :return:
    """
    fusion_time = fusion_list.timeStamp
    for fusion_obj in fusion_list:
        last_update = fusion_obj.last_update_time
        D = np.linalg.norm(fusion_obj.s_vector[:3])
        if fusion_time - last_update > 1. and D > distance_to_ego:  
            fusion_list.remove(fusion_obj)

    return fusion_list


debug = False
if debug:
    from objectClasses.objectClasses import Obstacle
    from objectClasses.objectClasses import Sensor, fusionList
    import numpy as np
    import matplotlib.pyplot as plt

    # create objects
    P_init = np.eye(11)
    fusion_obj = Obstacle(0, 0, 0, 0, 0, 0, None, None, 
                          None, None, None, P=np.eye(11))
    sensor_obj = Obstacle(0, 0, 0, 0, 0, 0, None, None, 
                          None, None, None, P=np.eye(11))
    fusion_list = fusionList(timeStamp=0)
    fusion_list.append(fusion_obj)
    sensor_list = fusionList(timeStamp=0)
    sensor_list.append(sensor_obj)

    sensor1 = Sensor(timeStamp=0, obj_list=sensor_list, 
                     H_sensor_veh=np.eye(11))

    # create true states
    vel_x, vel_y = 5., 3.
    true_states = np.zeros((50, 11))  # 50 samples
    true_vel_x = vel_x * np.ones((50,))  # const velocity
    true_pos_x = np.arange(0, 50 * vel_x, vel_x)  #
    true_vel_y = vel_y * np.ones((50,))  # const velocity
    true_pos_y = np.arange(0, 50 * vel_y, vel_y)  #

    true_states[:, 0] = true_pos_x
    true_states[:, 3] = true_vel_x
    true_states[:, 1] = true_pos_y
    true_states[:, 4] = true_vel_y

    # add measurements with noise
    measurements = np.empty((50, 11))
    measurements[:] = np.nan
    measurements_time = 3  # get measurements in every 3 secs
    measurements[::measurements_time, :] = true_states[::measurements_time, :]
    measurements[::measurements_time, np.isnan(sensor_obj.s_vector)] = np.nan

    predicted_state = []
    for idx, (true_state, measurement) in enumerate(zip(true_states, 
                                                        measurements)):
        if not np.isnan(measurement).all():
            sensor1.timeStamp = idx
            noise = 5. * np.random.normal(size=(11,))
            sensor_obj.s_vector = measurement + noise

            temporal_alignment(fusion_list, sensor1.timeStamp)
            kf_measurement_update(fusion_list, sensor1.obj_list, ((0, 0), (0, 0)))

        predicted_state.append(np.copy(fusion_obj.s_vector))

    predicted_state = np.array(predicted_state)

    fig, axs = plt.subplots(1, )

    l1 = axs.plot(predicted_state[:, 0], predicted_state[:, 1],
                  label='Predicted Position')
    l2 = axs.plot(true_states[:, 0], true_states[:, 1], label='True Position')

    axs.axis()
    axs.legend()
    fig.suptitle('Kalman Filter')
    fig.set_size_inches((7, 3))
    plt.show()
