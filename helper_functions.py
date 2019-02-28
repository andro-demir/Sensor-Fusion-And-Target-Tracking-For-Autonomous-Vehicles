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
            F = np.array([[1, 0, delta, 0,     0.5*delta**2, 0,            0, 0],
                          [0, 1, 0,     delta, 0,            0.5*delta**2, 0, 0],
                          [0, 0, 1,     0,     delta,        0,            0, 0],
                          [0, 0, 0,     1,     0,            delta,        0, 0],
                          [0, 0, 0,     0,     1,            0,            0, 0],
                          [0, 0, 0,     0,     0,            1,            0, 0],
                          [0, 0, 0,     0,     0,            0,            1, delta],
                          [0, 0, 0,     0,     0,            0,            0, 0]])

            w = np.zeros((8,))
            w[4:6] = np.random.normal(size=(2,))  # noise added to accelerations

            Q = np.zeros((8,8))
            Q[4:6, 4:6] = np.multiply(np.random.normal(size=(2,2)), np.eye(2))  # noise added only at the last derivatives:
            Q[-1, -1] = np.random.normal()
        else:  # z axis is included
            F = np.array([[1, 0, 0,     delta, 0,     0,     0.5*delta**2, 0, 0,   0, 0],
                          [0, 1, 0,     0,     delta, 0,     0, 0.5*delta**2, 0,   0, 0],
                          [0, 0, 1,     0,     0,     delta, 0, 0, 0.5*delta**2,   0, 0],
                          [0, 0, 0,     1,     0,     0,     delta, 0, 0,          0, 0],
                          [0, 0, 0,     0,     1,     0,     0, delta, 0,          0, 0],
                          [0, 0, 0,     0,     0,     1,     0, 0, delta,          0, 0],
                          [0, 0, 0,     0,     0,     0,                  1, 0, 0, 0, 0],
                          [0, 0, 0,     0,     0,     0,                  0, 1, 0, 0, 0],
                          [0, 0, 0,     0,     0,     0,                  0, 0, 1, 0, 0],
                          [0, 0, 0,     0,     0,     0,              0, 0, 0, 1, delta],
                          [0, 0, 0,     0,     0,     0, 0, 0, 0,                 0, 0]])

            w = np.zeros((11,))
            w[6:9] = np.random.normal(size=(3,))  # noise added to accelerations

            Q = np.zeros((11, 11))
            Q[6:9, 6:9] = np.multiply(np.random.normal(size=(3, 3)), np.eye(3))  # noise added only at the last derivatives:
            Q[-1, -1] = np.random.normal()

        obj.s_vector = np.dot(F, obj.s_vector) + obj.u + w
        obj.P = np.dot(np.dot(F, obj.P), F.T) + Q
        pass

    if method == 'SingleStep':
        for obj in obj_list:
            delta = current_time - obj.timeStamp  # update delta
            alignment_equations(obj, delta=delta)
        obj_list.timeStamp = current_time

    elif method == 'EqualStep':
        delta = 1  # ! TODO: is this a correct precision??
        for obj in obj_list:
            for _ in range(obj_list.timeStamp + delta, current_time + delta, delta):
                alignment_equations(obj, delta=delta)
        obj_list.timeStamp = current_time

    pass
