# Advanced Driver Assistance Systems (ADAS)

### Table of contents
1. [Introduction](### Introduction)
2. [Software Requirements](### Software Requirements)
3. [Data and Experiments](#data)
    1. [Demo](#demo)
    2. [Hyper-parameter Tuning](#hyperparameter)
4. [Contact](#contact)
    
### Introduction
This software package implements a low-level sensor data fusion algorithm, in which the data extracted from highly synchronized sensors is combined associating the radar and vision measurements and then feeding the fused measurements to a central tracking algorithm, based on Kalman filter updates. 

The first step is to temporally and spatially align the sensor-level object lists from all the sensors to a common reference frame. This puts all of the object list into a global coordinate system. Once this is accomplished, the object lists from all of the sensors are associated with one another in order to determine which object from different sensors correspond to the same object in reality. Combined with other perception modules such as lane detection, digital maps, host vehicle localization, this provides any driver assistance application specific situation assessment algorithms. Some examples of those are controlling an actuator, triggering a warning and changing a state.

### Software Requirements

Instructions to enable Python - Matlab Interoperability
```
1. Create a virtual environment
2. Install Python 3.6
3. Install Python modules:
    - numpy v. 1.16.2
    - scikit-learn v. 0.19.1
    - scipy v. 1.1.0
    - sklearn v. 0.0
4. Start Matlab and setup the Python interpreter for Matlab with: pyversion(path_to_python.exe_in_your_virtual_env)
```

### Data and Experiments

#### Demo
After successfully completing basic installation, you'll be ready to run the demo.

To run the demo:
```
Run SF_Synthetic_Main.m
```
![image](https://user-images.githubusercontent.com/43050657/57249662-88018000-7013-11e9-9cec-35bf6d646bab.png)
Arrows represent the velocity vectors of other actors in the scene.

#### Hyperparameter Tuning
On matlabDemo.py, you can tune the hyperparameters:
```
--clutter_threshold: type=float, default=0.75, help='if mahalanobis distance > clutter thresholod, assign the object as false positive'
--last_seen: type=float, default=1.0, help='if the tracked object has not been seen longer than last_seen, delete it from the fusion list'
--distance_to_ego: type=float, default=200, help='distance to ego (L1 norm of the tracked objects' state vector)'
```

#### Contact
ademir@ece.neu.edu
