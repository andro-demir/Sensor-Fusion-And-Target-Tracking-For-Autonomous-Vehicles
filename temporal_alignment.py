import numpy as np


def temporal_alignment(obj, current_time):
    """
    preliminary kalman filter update for the obj.
    :param obj: (class) the object beeing tracked with the properties: last_state, x(current state),
     F (state dynamics matrix), u (input), w (proces noise model), P (state covariance matrix), Q (cov of process noise)
    :param current_time:
    :return:
    """
    for time in range(obj.timeStamp, current_time):
        obj.last_state = obj.x
        obj.x = np.dot(obj.F, obj.x) + obj.u + obj.w
        obj.P = np.dot(np.dot(obj.F, obj.P), obj.F.T) + obj.Q
        obj.timeStamp = time
    return

debug = False

if debug:
    class object():  # dummy object class
        def __init__(self):
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
            self.u = np.zeros((8,))  # zeros for constant vel model, input should also change.
            self.w = np.random.random((8,))  # process noise
            Q = np.zeros((8,8))
            Q[3:5, 3:5] = np.random.random() * np.eye(2)  # noise added only at the last derivatives
            Q[7, 7] = np.random.random()
            self.Q = Q
            pass

    obj = object()

    obj_start_state = obj.x

    temporal_alignment(obj, 1)  # one step

    obj_state_1_iter = obj.x







