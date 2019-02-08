import argparse
from readData import readFromPath
from objectClasses import imu, lane, radar, vision 
from time import perf_counter

def parseArgs():
    parser = argparse.ArgumentParser(description='Multi-Target Tracking',
                                     formatter_class=argparse.
                                            ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--data', help=('Input file containing 4 structs: '
                        'radar measurements, vision measurements, lane '
                        'measurements and IMU measurements.'), type=str,
                        default="DATA/01_city_c2s_fcw_10s_sensor.mat")
    parser.add_argument('--output', help='File in which output is stored', 
                        type=str, default=None)
    parser.add_argument('--mode', help='Mode of operation', 
                        choices=['vision-radar'], default="vision-radar")
    args = parser.parse_args()
    return args
    
def main():
    args = parseArgs()
    data = readFromPath(args.data)
    print(data['radar']['timeStamp'])

if __name__ == "__main__":
    start = perf_counter()
    main()
    duration = perf_counter() - start
    print("Performance: %f secs" %duration)
