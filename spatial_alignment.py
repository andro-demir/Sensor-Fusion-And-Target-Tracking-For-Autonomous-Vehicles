import numpy as np

def spatial_alignment(sensor, object_id):
    """
    Transform the state vector of the object from sensor coordinate frame to vehicle coordinate frame

    :param sensor: (class) sensor class that contains a list of objects(class), H_sensor_veh transformation matrix
    :param object_id: index of the object in the object list of sensor
    :return: x: ((n,) array) state vector in the vehicle coordinate frame
             P: ((nxn) array) covariance matrix of the state x in the vehicle coordinate frame

    """
    obj = sensor.objects[object_id]
    x = np.dot(sensor.H_sensor_veh, np.concatenate((obj.x, [1])))[:-1]
    P = np.dot(np.dot(sensor.H_sensor_veh[:-1, :-1], obj.P), sensor.H_sensor_veh[:-1, :-1].T)

    return x, P


debug = False
if debug:
    class object:  # dummy object class
        def __init__(self):
            self.x = np.array((2,0,0,0,0,0,0.,0.))#np.random.random((8,))  # random state vector
            self.P = np.random.random((8, 8))  # random cov matrix
            return

    class sens:  # dummy example of a sensor class
        def __init__(self):

            self.psi = np.pi/3.  # angle between car and sensor view
            self.delta = np.array((1., np.sqrt(3)))  # position of the sensor wrt car center.
            rot_psi = np.array([[np.cos(self.psi), -np.sin(self.psi)], [np.sin(self.psi), np.cos(self.psi)]])
            H_sensor_veh = np.zeros((9, 9))
            for i in range(3): H_sensor_veh[i*2:(i+1)*2, i*2:(i+1)*2] = rot_psi  # fill the diagonal matricies
            H_sensor_veh[6:9, 6:9] = np.eye(3)
            H_sensor_veh[0:2,-1] = self.delta
            H_sensor_veh[6, -1] = self.psi

            self.H_sensor_veh = H_sensor_veh

            self.objects = [object() for _ in range(2)]

            return

    dummy_sensor = sens()

    object_id = 0

    object_1_state = dummy_sensor.objects[object_id].x

    object_1_state_tranformed, object_1_cov_transformed = spatial_alignment(dummy_sensor, object_id)















