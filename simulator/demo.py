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
    fusionList = fusionListCls(time_frame[0])
    fusion_hist = [i for i in fusionList]
    fusionList.extend(list_object_cam_front + list_object_cam_rear +
                      list_object_radar_front + list_object_radar_rear)
    for obstacle in fusionList:
        print("At Time: %f, State Vector:" % time_frame[0])
        print([x for x in obstacle.s_vector if x is not None])

    hebele = []
    # We created the fusionList at time,
    # Get the sensorObjectList at time+1
    for idx, time in enumerate(time_frame[:-1]):
        sensorObjList_at_Idx = []
        for sensor in [cam_front, cam_rear, radar_front, radar_rear]:
            list_object, _ = sensor.return_obstacle_list(time_frame[idx + 1])
            sensorObjList_at_Idx.extend(list_object)
            # Sensor data association
            sensorObjList = []
            sensorObjList.extend(list_object)
            mahalanobisMatrix = assc.getMahalanobisMatrix(fusionList, 
                                                          sensorObjList)
            rowInd, colInd = assc.matchObjs(mahalanobisMatrix)
            temporal_alignment(fusionList, time)
            kf_measurement_update(fusionList, sensorObjList, (rowInd, colInd))
            
            # Probability of existence of obstacles is updated:
            fusionList = assc.updateExistenceProbability(fusionList,
                                                         sensorObjList,
                                                         rowInd, colInd)
        
        hebele.append(fusionList[0].s_vector)
        print(50 * '**')
        print("At Time: %f" % time_frame[idx + 1])
        print("Fusion List")
        for obstacle in fusionList:
            print(obstacle.s_vector)
        print("Sensor List")
        for obstacle in sensorObjList_at_Idx:
            print(obstacle.s_vector)
        print("Mahalanobis matrix:\n", mahalanobisMatrix)
        print("Row indices:\n", rowInd)
        print("Column indices:\n", colInd)

        if idx == 2:
            exit(1)

        fusion_hist.append([i.s_vector for i in fusionList])

    def empty_states(r, c):
        states = np.empty((r, c))
        states.fill(np.nan)
        return states

    obj_states = [empty_states(len(fusion_hist), len(obstacle.s_vector)) for _ in
                  fusionList]

    for idx, s_vectors in enumerate(fusion_hist):
        for obj_idx, s_vector in enumerate(s_vectors):
            obj_states[obj_idx][idx] = np.copy(s_vector)

    sensor_measures = []
    for sensor in [cam_front, cam_rear, radar_front, radar_rear]:
        print (set(np.array(sensor.list_object_id, dtype=int)[:, 0, 0]))
        indicies = [np.where(np.array(sensor.list_object_id) == obj_id)[0] for obj_id in
                    set(np.array(sensor.list_object_id, dtype=int)[:, 0, 0])]
        sensor_measures.append([[sensor.list_state[i] for i in obj] for obj in indicies])

    return

def plot_sensor_measurements(sensor_measures):
    cmaps = ['Reds', 'Blues', 'Greys', 'Purples', 'Oranges', 'Greens']
    obj_marks = ['.', '*', 'o']
    fig, axs = plt.subplots()
    for sensor_idx, measurements in enumerate(sensor_measures[1:2]):
        for obj_idx, obj_measurements in enumerate(measurements):
            print(obj_idx)

            if obj_measurements:
                obj_measurements = np.array(obj_measurements)
                c = np.linspace(0, 1, len(obj_measurements))
                axs.scatter(obj_measurements[:, 1, 0], obj_measurements[:, 0, 0],
                            c=c, cmap=cmaps[sensor_idx], marker=obj_marks[obj_idx],
                            label='Sens %d, Obj %d' % (sensor_idx, obj_idx))

    axs.set_ylabel('Y')
    axs.set_xlabel('X')
    axs.set_ylim([-150, 150])
    axs.set_xlim([-150, 150])

    plt.legend()
    plt.show()

def scatter(obj_states):
    cmaps = ['Reds', 'Blues', 'Greys', 'Purples', 'Oranges', 'Greens']
    for i in range(6):
        all_states = obj_states[i]
        c = np.linspace(0, 1, len(all_states))
        fig, axs = plt.subplots()

        axs.scatter(all_states[:, 1], all_states[:, 0], label='Obj', c=c, cmap=cmaps[i])

        axs.set_ylabel('Y')
        axs.set_xlabel('X')
        axs.set_ylim([-150, 150])
        axs.set_xlim([-150, 150])

        # plt.legend()
        plt.show()


'''
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
