from numpy import asarray
import numpy as np
import scipy.io as sio


class Obstacle:
    """ Obstacle class for the environment. Represents an object which is
        observed (detected) by any of the sensors.
        Attr:
            s_vector(ndarray[float]): an array of the following properties
                pos_x/y(float): position on the respective axis
                v_x/y(float): velocity on the respective axis
                a_x/y(float): acceleration on the respective axis
                yaw(float): yaw angle
                r_yaw(float): yaw rate
            P(ndarray[float]):
            dim()
            dim_uncertainty():
            p_existence(float): probability of existence
            c()
            f()
            """

    def __init__(self, pos_x, pos_y, v_x, v_y, a_x, a_y, yaw, r_yaw, P=[],
                 dim=(0, 0), dim_uncertainty=0, p_existence=0, c=None, f=None):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.v_x = v_x
        self.v_y = v_y
        self.a_x = a_x
        self.a_y = a_y
        self.yaw = yaw
        self.r_yaw = r_yaw
        self.s_vector = asarray([pos_x, pos_y, v_x, v_y, a_x, a_y, yaw, r_yaw])
        self.P = P
        self.dim = dim
        self.dim_uncertainty = dim_uncertainty
        self.p_existence = p_existence
        self.c = c
        self.f = f


class SimSensor(object):
    """ Simulator sensor object. Different than actual sensor models, accepts a
        Matlab file with pre-defined variables. Passes the object list of
        required time.
        Attr:
            filename[str]: filename for the sensor model
            list_time[ndarray(float)]: list of timestamps of observed objects
            list_state[list[ndarray(float)]]: list of state vectors
            list_noise[list[ndarray(float)]]: list of state noise models
            list_object_id[list[ndarray(float)]]: list of objects observed
                all lists should have the same dimension. If a timestamp is
                duplicated this indicates the sensor observed two objects.
            """

    def __init__(self, filename):
        self.filename = filename
        tmp = sio.loadmat(filename)
        self.list_time = np.squeeze(list(tmp['list_time'][0]))
        self.list_state = list(tmp['list_state'][0])
        self.list_noise = list(tmp['list_noise'][0])
        self.list_object_id = list(tmp['list_obj'][0])

    def return_obstacle_list(self, time):
        """ Returns observed obstacles at a given time
            Args:
                time(float): Time should match with the exact time. For
                    simulation purposes this is not crucial
            Return:
                list_obstacle(list[Obstacle]): Observed obstacles by the sensor
                time(float)
                """
        time_idx = np.where(self.list_time == time)[0]
        list_obstacle = []

        for idx_obj in list(time_idx):
            tmp_state = self.list_state[idx_obj]
            tmp_noise = self.list_noise[idx_obj]

            list_obstacle.append(
                Obstacle(tmp_state[0], tmp_state[1], tmp_state[2],
                         tmp_state[3], tmp_state[4], tmp_state[5], 0, 0,
                         P=tmp_noise))

        return list_obstacle, time


# TODO: Sensor Class and Subclasses Require Edits...

class Sensor:
    def __init__(self):
        pass

    def spatialAlignment(self):
        pass


class Radar(Sensor):
    def __init__(self, timeStamp, obj, numObjects):
        Sensor.__init__(self)
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects

    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(Radar, self).spatialAlignment()


class Vision(Sensor):
    def __init__(self, timeStamp, obj, numObjects):
        Sensor.__init__(self)
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects

    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(Vision, self).spatialAlignment()


class Lane(Sensor):
    def __init__(self, left, right):
        Sensor.__init__(self)
        self.left = left
        self.right = right

    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(Lane, self).spatialAlignment()


class IMU(Sensor):
    def __init__(self, timeStamp, velocity, yawRate):
        Sensor.__init__(self)
        self.timeStamp = timeStamp
        self.velocity = velocity
        self.yawRate = yawRate

    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(IMU, self).spatialAlignment()
