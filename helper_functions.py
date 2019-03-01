import numpy as np


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

            Q = np.zeros((8, 8))
            Q[4:6, 4:6] = np.multiply(np.random.normal(size=(2, 2)), np.eye(
                2))  # noise added only at the last derivatives:
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
            w[6:9] = np.random.normal(size=(3,))  # noise added to accelerations

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


# TODO: following function is not complete yet

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
        not_nan_idx = np.where(np.invert(np.isnan(s_vector_s)))[0]

        # nan_idx_fus = np.where(np.isnan(s_vector_f))[0]
        # nan_idx_sens = np.where(np.isnan(s_vector_s))[0]
        #
        # not_nan_idx = np.array(list(set(range(s_vector_f.shape[0])) - set(
        #     np.concatenate((nan_idx_fus, nan_idx_sens)))))

        s_vector_s = s_vector_s[not_nan_idx]
        s_vector_f = np.copy(fusion_obj.s_vector)[not_nan_idx]

        P_f = np.copy(fusion_obj.P)[not_nan_idx, :][:, not_nan_idx]
        P_s = np.copy(sensor_obj.P)[not_nan_idx, :][:, not_nan_idx]
        H = sensor_obj.H[not_nan_idx, :][:, not_nan_idx]

        # kalman filter equations:
        # R is the cov of the obs noise
        S = np.dot(np.dot(H, P_f), H.T) + P_s

        # K is the kalman gain
        K = np.dot(np.dot(P_f, H.T), np.linalg.inv(S))
        # Updated aposteriori state estimate
        x = s_vector_f + np.dot(K, s_vector_s - np.dot(H, s_vector_f))

        # Updated aposteriori estimate covariance
        P = np.dot(np.eye(s_vector_f.shape[0]) - np.dot(K, H), P_f)
        # update global object state and covariance

        # fusion_obj.P[np.where(sensor_obj.P != None)] = P[
        #     np.where(sensor_obj.P != None)]
        #
        # fusion_obj.s_vector[np.where(sensor_obj.s_vector != None)] = x[
        #     np.where(sensor_obj.s_vector != None)]

        fusion_obj.P[not_nan_idx, :][:, not_nan_idx] = P

        fusion_obj.s_vector[not_nan_idx] = x

    pass


debug = False
if debug:
    from objectClasses.objectClasses import Obstacle
    from objectClasses.objectClasses import Sensor, fusionList
    import numpy as np
    import matplotlib.pyplot as plt

    # create objects
    P_init = np.eye(11)
    fusion_obj = Obstacle(0, 0, 0, 0, 0, 0, None, None, None, None, None, P=np.eye(11))
    sensor_obj = Obstacle(0, 0, 0, 0, 0, 0, None, None, None, None, None, P=np.eye(11))
    fusion_list = fusionList(timeStamp=0)
    fusion_list.append(fusion_obj)
    sensor_list = fusionList(timeStamp=0)
    sensor_list.append(sensor_obj)

    sensor1 = Sensor(timeStamp=0, obj_list=sensor_list, H_sensor_veh=np.eye(11))

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
    for idx, (true_state, measurement) in enumerate(zip(true_states, measurements)):
        if not np.isnan(measurement).all():
            sensor1.timeStamp = idx
            noise = 5. * np.random.normal(size=(11,))
            # noise[np.where(sensor_obj.s_vector == None)] = None
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
