from temporalAlignment import *
from kalmanFilter import *
import numpy as np
import matplotlib.pyplot as plt

class object():  # dummy object class
    def __init__(self, is_sensor=False):
        self.timeStamp = 0
        self.x = np.zeros((8,))  # random state
        self.P = 0.1 * np.random.normal() * np.eye(8)    # random cov
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
        self.w = 0.1 * np.random.normal(size=(8,))  # process noise
        Q = np.zeros((8,8))
        Q[3:5, 3:5] = 0.1 * np.random.normal() * np.eye(2)  # noise added only at the last derivatives
        Q[7, 7] = 0.1 * np.random.normal()
        self.Q = Q

        if is_sensor:  # add the other params for sensor: this is a dummy example normally sensor and fusion object might be different
            self.R = 0.1 * np.random.normal() * np.eye(8)  # measurement noise covariance
            self.H = np.eye(8)
        pass

    def __setattr__(self, key, value):
        if key == 'delta':  # F is delta dependent when delta is updated F should be updated too
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

        pass

# create objects
fusion_obj = object()
sensor_obj = object(is_sensor=True)

# create true states
vel_x, vel_y = 5., 3.
true_states = np.zeros((50,8))  # 50 samples
true_vel_x = vel_x * np.ones((50,))  # const velocity
true_pos_x = np.arange(0,50 * vel_x, vel_x)  #
true_vel_y = vel_y * np.ones((50,))  # const velocity
true_pos_y = np.arange(0,50 * vel_y, vel_y)  #

true_states[:, 0] = true_pos_x
true_states[:, 2] = true_vel_x
true_states[:, 1] = true_pos_y
true_states[:, 3] = true_vel_y

# add measurements with noise
measurements = np.empty((50, 8))
measurements[:] = np.nan
measurements_time = 3  # get measurements in every 3 secs
measurements[::measurements_time, :] = true_states[::measurements_time, :]
predicted_state = []
for idx, (true_state, measurement) in enumerate(zip(true_states,measurements)):
    if not np.isnan(measurement).any():
        sensor_obj.timeStamp = idx
        sensor_obj.x = measurement + 5. * np.random.normal(size=(8,))# noise added

        temporal_alignment(fusion_obj, sensor_obj.timeStamp)
        kf_measurement_update(fusion_obj,sensor_obj)

    predicted_state.append(fusion_obj.x)

predicted_state = np.array(predicted_state)


fig, axs = plt.subplots(1,)

l1 = axs.plot(predicted_state[:, 0], predicted_state[:, 1], label='Predicted Position')
l2 = axs.plot(true_states[:, 0], true_states[:, 1], label='True Position')

axs.axis()
axs.legend()
fig.suptitle('Kalman Filter')
fig.set_size_inches((7,3))
plt.show()



