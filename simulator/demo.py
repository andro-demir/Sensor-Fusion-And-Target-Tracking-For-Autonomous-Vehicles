import sys

sys.path.append("..")
import argparse
import numpy as np
from objectClasses.objectClasses import SimSensor
import objectAssociation as assc
from time import perf_counter
import warnings
from objectClasses.objectClasses import fusionList as fusionListCls
from time import perf_counter
from helper_functions import kf_measurement_update, temporal_alignment
import matplotlib.pyplot as plt


def createSensorEnvs():
    '''
    Create environment sensors using Matlab file.
    '''
    cam_rear = SimSensor('./cam_rear.mat')
    cam_front = SimSensor('./cam_front.mat')
    radar_rear = SimSensor('./radar_rear.mat')
    radar_front = SimSensor('./radar_front.mat')
    return cam_rear, cam_front, radar_rear, radar_front


def main():
    cam_rear, cam_front, radar_rear, radar_front = createSensorEnvs()
    time_frame = [cam_rear.list_time, cam_front.list_time,
                  radar_rear.list_time, radar_front.list_time]

    # Create the time frame from unique entities of the times recorded for each
    # sensor. This will serve as the time scale for the rest of the system.
    time_frame = list(np.unique(np.concatenate(time_frame)))

    # Initialize the fusion list:
    list_object_cam_front, _ = cam_front.return_obstacle_list(time_frame[0])
    list_object_cam_rear, _ = cam_rear.return_obstacle_list(time_frame[0])
    list_object_radar_front, _ = radar_front.return_obstacle_list(time_frame[0])
    list_object_radar_rear, _ = radar_rear.return_obstacle_list(time_frame[0])
    fusionList = (list_object_cam_front + list_object_cam_rear +
                  list_object_radar_front + list_object_radar_rear) 
    for obstacle in fusionList:
        print("At Time: %f, State Vector:" %time_frame[0])
        print([x for x in obstacle.s_vector if x is not None])
    
    # We created the fusionList at time,
    # Get the sensorObjectList at time+1
    for idx, _ in enumerate(time_frame[:-1]):
        list_object_cam_front, _ = cam_front.return_obstacle_list(
                                                time_frame[idx+1])
        list_object_cam_rear, _ = cam_rear.return_obstacle_list(
                                              time_frame[idx+1])
        list_object_radar_front, _ = radar_front.return_obstacle_list(
                                                    time_frame[idx+1])
        list_object_radar_rear, _ = radar_rear.return_obstacle_list(
                                                  time_frame[idx+1])
        
        # Sensor data association
        sensorObjList = (list_object_cam_front + list_object_cam_rear + list_object_radar_front + list_object_radar_rear)
        mahalanobisMatrix = assc.getMahalanobisMatrix(fusionList,  sensorObjList)
        rowInd, colInd = assc.matchObjs(mahalanobisMatrix)
        # Probability of existence of obstacles is updated:
        fusionList = assc.updateExistenceProbability(fusionList, 
                        sensorObjList, mahalanobisMatrix, rowInd, colInd)
        print(20 * '-')
        print("At Time: %f" %time_frame[idx+1])
        print("Mahalanobis matrix:\n", mahalanobisMatrix)
        print("Row indices:\n", rowInd)
        print("Column indices:\n", colInd)
        print("State Vector(s):")
        for obstacle in fusionList:
            print(obstacle.s_vector)

        if idx == 10:
            exit(1)
        # TODO:
        # Veysi's part (Fusion update):
'''      
    list_object_cam_rear, _, obj_ids_cam_rear = cam_rear.return_obstacle_list(
        time_frame[0])
    list_object_cam_front, _, obj_ids_cam_front = cam_front.return_obstacle_list(
        time_frame[0])
    list_object_radar_front, _, obj_ids_radar_front = radar_front.return_obstacle_list(
        time_frame[0])
    list_object_radar_rear, _, obj_ids_radar_rear = radar_rear.return_obstacle_list(
        time_frame[0])

    fusionList = fusionListCls(time_frame[0])
    fusionList.extend(list_object_radar_front)

    tracked_object_id = 0
    fusion_object_states = []
    measured = False
    for time in time_frame:
        list_object_cam_rear, _, obj_ids_cam_rear = cam_rear.return_obstacle_list(time)
        list_object_cam_front, _, obj_ids_cam_front = cam_front.return_obstacle_list(
            time)
        list_object_radar_front, _, obj_ids_radar_front = radar_front.return_obstacle_list(
            time)
        list_object_radar_rear, _, obj_ids_radar_rear = radar_rear.return_obstacle_list(
            time)

        # Sensor data association
        for sensor_idx, sensorObjList in enumerate(
                [list_object_radar_front]):
            for obj_idx, obj in enumerate(sensorObjList):
                if obj_ids_radar_front[obj_idx] == tracked_object_id:
                    print('Got Measurement')
                    measured = True
                    temporal_alignment(fusionList, time)
                    kf_measurement_update(fusionList, [obj], ([0], [0]))

                    fusion_object_states.append(np.copy(fusionList[0].s_vector))

    fusion_object_states = np.array(fusion_object_states)

    # plot([radar_front], fusion_object_states, which_sensor_idx=0,
    #      which_object=tracked_object_id)

    #
    # assc = Association(fusionList, sensorObjList)
    # assc.updateExistenceProbability()
    # # to get the H matrix call assc.rowInd and assc.colInd at each iter
    # # (You might need this when you do fusion)
    #
    # # to update the fusion list:
    # fusionList = assc.fusionList
    # for obstacle in fusionList:
    #     print("Time: %f, State Vector:" %time)
    #     print(obstacle.s_vector)
    # radar_rear_measurements = radar_rear.list_state[
    #     np.where(np.array(radar_rear.list_object_id) == tracked_object_id)[0]]
    print('done')


def plot(sensors, predicted_states, which_sensor_idx=0, which_object=0):
    """
    plot the measurements for each object and plot the predictions
    :return:
    """
    working_sensor = sensors[which_sensor_idx]
    obj_idx = [idx for idx, id in enumerate(working_sensor.list_object_id) if
               id == which_object]
    measured_states = [working_sensor.list_state[idx] for idx in obj_idx]
    measured_states = np.array(measured_states).reshape(len(measured_states), 6)

    if len(predicted_states) < len(measured_states):
        sys.exit('Predictions are not provided')

    fig, axs = plt.subplots()
    axs.plot(measured_states[:, 0], label='Measured X', c='b', linestyle=':')
    axs.plot(measured_states[:, 1], label='Measured Y', c='b', linestyle='-')
    axs.plot(predicted_states[:, 0], label='Predicted X', c='r', linestyle=':')
    axs.plot(predicted_states[:, 1], label='Predicted Y', c='r', linestyle='-')
    axs.set_ylabel('Value')
    axs.set_xlabel('Time')
    plt.suptitle(working_sensor.name_sensor + ', obj ' + str(which_object))
    plt.legend()
    plt.show()

    pass
'''

if __name__ == "__main__":
    start = perf_counter()
    main()
    duration = perf_counter() - start
    print("Performance: %f secs" % duration)
