# Advanced Driver-Assistance Systems (ADAS)

### Disclaimer
This program is not free software; you cannot redistribute it and/or modify it.

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

### Demo
After successfully completing basic installation, you'll be ready to run the demo.

To run the demo:
```
Run SF_Synthetic_Main.m
```
![image](https://user-images.githubusercontent.com/43050657/57249662-88018000-7013-11e9-9cec-35bf6d646bab.png)
Arrows represent the velocity vectors of other actors in the scene.

### Hyperparameter Tuning
On matlabDemo.py, you can tune the hyperparameters:
```
--clutter_threshold: type=float, default=0.75, help='if mahalanobis distance > clutter thresholod, assign the object as false positive'
--last_seen: type=float, default=1.0, help='if the tracked object has not been seen longer than last_seen, delete it from the fusion list'
--distance_to_ego: type=float, default=200, help='distance to ego (L1 norm of the tracked objects' state vector)'
```

### Contact
ademir@ece.neu.edu
