import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from Shared.PiController import PiController
import time

LC_DELAY_TIME = 40 #in secs

PI_HOST = "192.168.6.2" #IP of PI controlling DACs
USERNAME = "carillon"         
PASSWORD = "password"          
KEY_FILE = None # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"
PI_PORT = 22

STOP_FILE = r"/home/carillon/Downloads/STOP.txt"
LOCAL_FILE_HB = r"/home/carillon/Documents/datahb.txt" #HB voltage file to send to PI
LOCAL_FILE_LB = r"/home/carillon/Documents/datalb.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = r"/home/carillon/Desktop/DataHB.txt"  # where to put it on the Pi
REMOTE_FILE_LB = r"/home/carillon/Desktop/DataLB.txt"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/carillon/Gen3DAC60096EVM_SPI_RPi5_schollV2.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"nohup python3 {REMOTE_PROGRAM} >/dev/null 2>&1 &"


    
def update_lb_array_file(V, dual_band = False):
    #V = np.round(V * DAC_MIN_STEP_SIZE, 3)
    with open(LOCAL_FILE_LB, "w") as f:
        for row in V:
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")
    if dual_band == False:
        #This program is for LB only, so create a 0V array for the high band which is 24x8 in Sam's code
        with open(LOCAL_FILE_HB, "w") as f:
            for row in np.zeros((24,8)):
                line = ",".join(str(x) for x in row)
                f.write(line + "\n")
            
def update_hb_array_file(V, dual_band = False):
    #V = np.round(V * DAC_MIN_STEP_SIZE, 3)
    if dual_band == False:
        #This program is for HB only, so create a 0V array for the low band
        with open(LOCAL_FILE_LB, "w") as f:
            for row in np.zeros((12,8)):
                line = ",".join(str(x) for x in row)
                f.write(line + "\n")
    with open(LOCAL_FILE_HB, "w") as f:
        for row in V:
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")
def main():
    
    rpi = PiController(
        host=PI_HOST,
        username=USERNAME,
        password=PASSWORD,
        local_file_hb=LOCAL_FILE_HB,
        local_file_lb=LOCAL_FILE_LB,
        remote_file_hb=REMOTE_FILE_HB,
        remote_file_lb=REMOTE_FILE_LB,
        remote_command=REMOTE_COMMAND,
        port = PI_PORT,
        key_filename=KEY_FILE,
        stop_file = STOP_FILE,
    )
    rpi.connect()
    '''
    voltages= np.array([
    [1.11103, 2.07747, 4.44963, 10.50000, 0.00000, 1.52362, 3.55497, 10.50000],
    [1.41815, 3.20992, 10.50000, 0.00000, 1.30698, 2.75038, 10.50000, 0.00000],
    [0.31626, 2.24564, 8.40103, 0.00000, 1.02212, 2.23965, 6.38328, 0.00000],
    [1.45770, 4.81089, 10.50000, 0.59264, 1.81207, 3.88447, 10.50000, 0.42402],
    [0.36301, 2.69170, 10.50000, 0.00000, 1.37982, 2.62605, 8.24972, 0.00000],
    [1.71069, 6.67968, 0.00000, 0.84195, 1.93361, 4.19646, 10.50000, 0.74523],
    [0.77203, 3.12103, 10.50000, 0.00000, 1.34590, 2.50597, 7.58607, 0.00000],
    [1.87234, 6.59449, 0.00000, 0.59199, 1.70732, 3.37595, 10.50000, 0.47303],
    [0.96714, 2.81839, 10.50000, 0.00000, 0.95532, 2.04683, 5.10142, 10.50000],
    [1.61929, 3.78821, 10.50000, 0.00000, 1.22133, 2.43451, 7.43496, 0.00000],
    [0.55848, 1.83685, 4.19987, 10.50000, 0.00000, 1.47013, 2.85810, 8.53369],
    [0.55096, 1.74073, 4.07090, 10.50000, 0.34624, 1.69141, 2.94388, 5.32943],
])
'''
    voltages = np.loadtxt("/home/carillon/Documents/raopt-config/output/voltages_theta20_phi0.csv", delimiter=",")   
    update_lb_array_file(voltages, dual_band = False)
    
    rpi.update_dacs()
    time.sleep(40)
    print("Ready to run scan")
    #Manually stop run when scan is finish
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("exiting")
            break
    
    #Set array elements to zero when done
    v_close = np.round(np.zeros((12,8)),3)
    update_lb_array_file(v_close)
    rpi.update_dacs()
    rpi.stop_program()
    rpi.close()

if __name__ == "__main__":
    main()
