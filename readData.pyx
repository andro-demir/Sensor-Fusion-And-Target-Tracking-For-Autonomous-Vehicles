'''
****************************************************************
Contributors to this code
** Andac Demir (andacdemir@gmail.com) (main developer)
****************************************************************
'''
import numpy as np
import scipy.io as spio

def readFromPath(file_name):
    data = spio.loadmat(file_name, mdict=None, appendmat=True)
    return data

'''
    Short guide to how to read and call the data
'''
def main():
    fileName = ("DATA/01_city_c2s_fcw_10s_sensor.mat")
    data = readFromPath(fileName)
    assert type(data) == dict 
    # Keys: radar, vision, lane, inertialMeasurementUnit
    print("Keys of the data dict:\n", data.keys())
    # 1D array of 204 rows
    print(data['radar']['timeStamp'])
    # 2D array: 204 x 7, columns: id, status, position, velocity,
    #                             amplitude, rangeMode, rangeRate
    print(data['radar']['object'])
    # 1D array of 204 rows
    print(data['radar']['numObjects'])


if __name__ == "__main__":
    main()
