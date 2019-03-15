import sys

sys.path.append("..")
import argparse
import numpy as np
from objectClasses.objectClasses import SimSensor
import objectAssociation as assc
from time import perf_counter
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
          

if __name__ == "__main__":
    start = perf_counter()
    main()
    duration = perf_counter() - start
    print("Performance: %f secs" % duration)
