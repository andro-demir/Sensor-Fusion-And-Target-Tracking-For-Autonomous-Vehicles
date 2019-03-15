from numpy import asarray
import numpy as np
import scipy.io as sio
from helper_functions import *


class Obstacle:
    """ Obstacle class for the environment. Represents an object which is
        observed (detected) by any of the sensors.
        Attr:
            s_vector(ndarray[float]): an array of the following properties
                pos_x/y/z(float): position on the respective axis
                v_x/y/z(float): velocity on the respective axis
                a_x/y/z(float): acceleration on the respective axis
                yaw(float): yaw angle
                r_yaw(float): yaw rate
            P(ndarray[float]):
            dim()
            dim_uncertainty():
            p_existence(float): probability of existence
            c()
            f()
            """

    def __init__(self, pos_x, pos_y, pos_z, v_x, v_y, v_z, a_x, a_y, a_z, yaw,
                 r_yaw,
                 P=[], dim=(0, 0), dim_uncertainty=0, p_existence=0, c=None,
                 f=None):
        self.s_vector = asarray([pos_x, pos_y, pos_z,
                                 v_x, v_y, v_z, a_x, a_y, a_z, yaw, r_yaw]).astype(float)
        self.P = P
        self.dim = dim
        self.dim_uncertainty = dim_uncertainty
        self.p_existence = p_existence
        self.c = c
        self.f = f
        self.H = np.eye(self.s_vector.shape[0])
        # self.create_observation_matrix()
        self.u = np.zeros(shape=(self.s_vector.shape[0],))  # zero input model

    def create_observation_matrix(self):
        H = np.eye(self.s_vector.shape[0])
        # H[np.where(self.s_vector == None), np.where(self.s_vector == None)] = 0.
        H[np.isnan(self.s_vector), np.isnan(self.s_vector)] = 0.
        self.H = H
        pass


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
        self.name_sensor = filename.split('.')[1]

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
        ids_obstacle = []
        for idx_obj in list(time_idx):
            tmp_state = self.list_state[idx_obj]
            tmp_noise = self.list_noise[idx_obj]
            
            list_obstacle.append(
                Obstacle(pos_x=tmp_state[0][0], pos_y=tmp_state[1][0],
                         pos_z=tmp_state[2][0], v_x=tmp_state[3][0], 
                         v_y=tmp_state[4][0], v_z=tmp_state[5][0], 
                         a_x=None, a_y=None, a_z=None,
                         yaw=None, r_yaw=None, P=tmp_noise))
            ids_obstacle.append(self.list_object_id[idx_obj])
        return list_obstacle, time  # , ids_obstacle


# TODO: Sensor Class and Subclasses Require Edits...

class Sensor:
    def __init__(self, timeStamp, obj_list, H_sensor_veh=None):
        self.timeStamp = timeStamp
        self.obj_list = obj_list
        self.H_sensor_veh = H_sensor_veh
        pass

    def spatialAlignment(self):
        spatial_alignment(self.obj_list,
                          self.H_sensor_veh)  # from helper functions
        pass


class Radar(Sensor):
    def __init__(self, timeStamp, obj_list, H_sensor_veh):
        Sensor.__init__(self, timeStamp, obj_list, H_sensor_veh)


class Vision(Sensor):
    def __init__(self, timeStamp, obj_list, H_sensor_veh):
        Sensor.__init__(self, timeStamp, obj_list, H_sensor_veh)


class Lane(Sensor):
    def __init__(self, left, right,
                 time_stamp):  # TODO: not sure if sensor to veh transformation matrix should be incl
        Sensor.__init__(self, time_stamp, None)
        self.left = left
        self.right = right


class IMU(Sensor):
    def __init__(self, timeStamp, obj_list, velocity, yaw_rate, H_sensor_veh):
        Sensor.__init__(self, timeStamp, obj_list, H_sensor_veh)

        # TODO: what are these? why only in IMU?
        self.velocity = velocity
        self.yaw_rate = yaw_rate


class fusionList(list):
    def __init__(self, timeStamp):
        self.timeStamp = timeStamp
