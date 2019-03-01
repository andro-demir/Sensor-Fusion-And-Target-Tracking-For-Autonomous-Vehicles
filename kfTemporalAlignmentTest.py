from objectClasses.objectClasses import Obstacle
from objectClasses.objectClasses import Sensor, fusionList
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import temporal_alignment, kf_measurement_update

# create objects
P_init = np.eye(11)
fusion_obj = Obstacle(0, 0, 0, 0, 0, 0, None, None, None, None, None, P=np.eye(11))
sensor_obj = Obstacle(0, 0, 0, 0, 0, 0, None, None, None, None, None, P=np.eye(11))
fusion_list = fusionList(timeStamp=0)
fusion_list.append(fusion_obj)
sensor_list = fusionList(timeStamp=0)
sensor_list.append(sensor_obj)

sensor1 = Sensor(timeStamp=0, obj_list=sensor_list, H_sensor_veh=np.eye(11))

number_of_samples = 50
# create true states
vel_x, vel_y = 5., 3.
vals = np.arange(0,number_of_samples)
true_states = np.zeros((number_of_samples, 11))  # number_of_samples samples
true_vel_x = vel_x * np.ones((number_of_samples,))  # const velocity
true_pos_x = np.arange(0, number_of_samples * vel_x, vel_x)  #
true_vel_y = vel_y * np.ones((number_of_samples,))  # const velocity
true_pos_y = np.arange(0, number_of_samples * vel_y, vel_y)  #

true_states[:, 0] = true_pos_x
true_states[:, 3] = true_vel_x
true_states[:, 1] = true_pos_y
true_states[:, 4] = true_vel_y

# add measurements with noise
measurements = np.empty((number_of_samples, 11))
measurements[:] = np.nan
measurements_time = 3  # get measurements in every 3 secs
measurements[::measurements_time, :] = true_states[::measurements_time, :]
measurements[::measurements_time, np.isnan(sensor_obj.s_vector)] = np.nan

predicted_state = []
for idx, (true_state, measurement) in enumerate(zip(true_states, measurements)):
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
