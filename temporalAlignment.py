import numpy as np

def temporal_alignment(obj, current_time, method='SingleStep'):
    """
    preliminary kalman filter update for the obj.
    :param obj: (class) the object beeing tracked with the properties:
                last_state, x(current state),
                F (state dynamics matrix), u (input), 
                w (proces noise model), 
                P (state covariance matrix), Q (cov of process noise)
    :param current_time:
    :param method: (str) Method for integral, if 'SingleStep' single integral 
                   will be taken with delta equal to time
                   difference, if 'EqualStep' for every unit between 
                   current time and object time one integral will be taken.
    :return:
    """
    def alignment_equations(obj, time):
        obj.last_state = obj.x
        obj.last_P = obj.P
        obj.x = np.dot(obj.F, obj.x) + obj.u + obj.w
        obj.P = np.dot(np.dot(obj.F, obj.P), obj.F.T) + obj.Q
        obj.timeStamp = time
        pass

    if method == 'SingleStep':
        obj.delta = current_time - obj.timeStamp  # update delta
        alignment_equations(obj, time=current_time)

    elif method == 'EqualStep':
        delta = 1  # !
        obj.delta = delta
        for time in range(obj.timeStamp + delta, current_time + delta, delta):
            alignment_equations(obj, time=time)
        pass
    pass

debug = False

if debug:
    class object():  # dummy object class
        def __init__(self):
            self.timeStamp = 0
            self.x = np.random.random((8,))  # random state
            self.P = np.random.random((8,8))  # random cov
            self.delta = 1  # !
            self.F = np.array([[1, 0, self.delta, 0, 0.5*self.delta**2, 0, 0, 0],
                              [0, 1, 0, self.delta, 0, 0.5*self.delta**2, 0, 0],
                              [0, 0, 1, 0, self.delta, 0, 0, 0],
                              [0, 0, 0, 1, 0, self.delta, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, self.delta],
                              [0, 0, 0, 0, 0, 0, 0, 0]])
            # zeros for constant vel model, input should also change:
            self.u = np.zeros((8,))  
            self.w = np.random.random((8,))  # process noise
            Q = np.zeros((8,8))
            # noise added only at the last derivatives:
            Q[3:5, 3:5] = np.random.random() * np.eye(2)  
            Q[7, 7] = np.random.random()
            self.Q = Q
            pass

        def __setattr__(self, key, value):
            # F is delta dependent when delta is updated F should be updated too
            if key == 'delta':  
                super(object, self).__setattr__(key, value)
                F = np.array([[1, 0, self.delta, 0, 0.5 * self.delta ** 2, 0, 0, 0],
                              [0, 1, 0, self.delta, 0, 0.5 * self.delta ** 2, 0, 0],
                              [0, 0, 1, 0, self.delta, 0, 0, 0],
                              [0, 0, 0, 1, 0, self.delta, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, self.delta],
                              [0, 0, 0, 0, 0, 0, 0, 0]])
                super(object, self).__setattr__('F', F)
            else: super(object, self).__setattr__(key, value)

            # self.key = value


    obj = object()
    obj_start_state = obj.x
    temporal_alignment(obj, 4)  # one step
    obj_state_1_iter = obj.x







