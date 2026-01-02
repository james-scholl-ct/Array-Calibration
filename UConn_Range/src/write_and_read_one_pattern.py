# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:17 2025

@author: SchollJamesAC3CARILL


Get latest for first time with git clone https://github.com/james-scholl-ct/Array-Calibration.git then use git pull
Install packages from Array-Calibration folder with pip install -e . (Use spyder console if not installed)
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import subprocess
from Shared.PiController import PiController
from Shared.NSI2000Client import NSI2000Client

#Place to store experiment results
EXP_DIR = r"C:\NSI2000\Data\Carillon\reflectarray_calibration\Experiments"

SCAN_FILENAME = r"C:\NSI2000\Data\Carillon\calibration_scan_real.nsi"

#PHASE_MAP_FILE_LB = r"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\phases_with_beam_steering_0theta_0phi_hex_12x8.txt" 

PI_HOST = "192.168.6.30" #IP of PI controlling DACs
USERNAME = "feix"         
PASSWORD = "password"          
KEY_FILE = None # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"
PI_PORT = 22

STOP_FILE = r"/home/feix/STOP.txt"
LOCAL_FILE_HB = r"C:\NSI2000\Data\Carillon\HB_voltages.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = r"C:\NSI2000\Data\Carillon\LB_voltages.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = r"/home/feix/Desktop/dataHB.csv"  # where to put it on the Pi
REMOTE_FILE_LB = r"/home/feix/Downloads/2025-12-18 VoltageMap_HornCorrection.csv"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/feix/Gen3DAC60096EVM_SPI_RPi5_scholl.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"python3 {REMOTE_PROGRAM}"

INIT_VOLTAGE_MAP = np.array([
       [3.3, 3.52, 3.6, 3.69, 3.44, 3.28, 2.98, 2.79], 
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
       [0.37, 1.14, 1.21, 1.23, 1.09, 0.0, 0.0, 0.0], 
       [1.91, 1.98, 2.04, 2.05, 2.02, 1.97, 1.88, 1.79],
       [2.67, 2.79, 2.81, 2.83, 2.74, 2.66, 2.49, 2.35], 
       [5.17, 8.39, 0.0, 0.0, 1.12, 7.24, 4.73, 3.86], 
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
       [1.21, 1.42, 1.52, 1.53, 1.51, 1.38, 1.13, 0.0], 
       [1.98, 2.06, 2.08, 2.09, 2.03, 1.97, 1.87, 1.73], 
       [2.52, 2.64, 2.72, 2.73, 2.7, 2.62, 2.49, 2.36], 
       [3.91, 4.53, 4.74, 4.88, 4.32, 3.88, 3.38, 2.98], 
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
], dtype=float)
  
#Size of array representing elements on the board-12x8 for LB
SIZE = (12,8)

LC_DELAY_TIME = 40 #in secs

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

#Set the beam (corresponds to frequency measured) number that you put in NSI software. 19.3 Ghz is ideal for low band
BEAM = 27

ELEVATION = 0
AZIMUTH = 0

FREQUENCY = "19.3 Ghz"

#Loss function params
MAIN_LOBE_HALF_WIDTH = 2 #Number of points in scan for the main lobe half width
CENTER_INDEX = 15 #Index where the center lobe should be
GUARD_BAND_HALF_WIDTH = 4 #Number of points in the scan for a guard band not considered in loss function
 

# SPSA hyperparameters
a0 = 9000000   # learning-rate scale in dac steps
c0 = 600  # perturbation scale in DAC steps should be 2-5x a0
alpha = 0.6 #.6-.8
gamma = 0.1
num_iters = 200

def read_phase_map_file(filename):
    phasemap = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            phasemap.append([int(x) for x in line.split(",")])
    phasemap = np.flipud((np.array(phasemap).T))#transpose then flip up/down sams spi code takes 12x8 but current phasemap code gives 8x12
    return phasemap

def update_lb_array_file(V):
    #V = np.round(V * DAC_MIN_STEP_SIZE, 3)
    with open(LOCAL_FILE_LB, "w") as f:
        for row in V:
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")
    #This program is for LB only, so create a 0V array for the high band which is 24x8 in Sam's code
    with open(LOCAL_FILE_HB, "w") as f:
        for row in np.zeros((24,8)):
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")
    

def loss_center_vs_sidelobes_db(
    amp_db,
    center_idx: int,
    main_half_width: int = 2,
    guard_half_width: int = 10,
    loss_equation: int =  0,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-15,
):
    """
    SPSA-friendly scalar loss using VNA amplitude in dB (dB magnitude).

    Goal: maximize energy near known center_idx, minimize energy elsewhere (sidelobes).

    loss = 1 - E_main / (E_main + E_side) or
    loss = -E_main - lm * E_side

    - E_main: sum of linear power in [center-main_half_width, center+main_half_width]
    - E_side: sum of linear power outside a guard band
              [center-guard_half_width, center+guard_half_width]

    Returns
    -------
    loss : float
    """
    amp_db = np.asarray(amp_db, dtype=float)
    N = amp_db.size
    c = int(np.clip(center_idx, 0, N - 1))

    if guard_half_width < main_half_width:
        guard_half_width = main_half_width

    # dB magnitude -> linear power
    # If amp_db is dB magnitude: mag_lin = 10^(dB/20), power = mag_lin^2 = 10^(dB/10)
    pwr = 10.0 ** (amp_db / 10.0)

    # main window
    m0 = max(0, c - main_half_width)
    m1 = min(N, c + main_half_width + 1)

    # guard band (excluded from sidelobe calculation)
    g0 = max(0, c - guard_half_width)
    g1 = min(N, c + guard_half_width + 1)

    main_mask = np.zeros(N, dtype=bool)
    main_mask[m0:m1] = True

    side_mask = np.ones(N, dtype=bool)
    #side_mask[g0:g1] = False  # everything outside guard is "sidelobes"
    side_mask[g0:] = False  # everything outside guard is including all positive indices which are closest to horn (from milad- it interferes)
    
    E_main = float(np.sum(pwr[main_mask]))
    E_side = float(np.sum(pwr[side_mask]))

    loss = 1 - E_main / (E_main + E_side)
    
    #lm = .25
    #loss = -E_main - lm * E_side
    
    return loss

def compute_loss(v, vna_instance, rpi, k, is_loss_plus, cal_folder):
    """
    """
    #updates low band array file on local computer
    update_lb_array_file(v)
    
    #sends low and high band array files to PI and runs remote command to update DACs
    rpi.update_dacs()
    
    time.sleep(LC_DELAY_TIME)
    
    pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)
    vna_instance.save_scan(k, is_loss_plus, cal_folder)
    
    loss = loss_center_vs_sidelobes_db(pattern, CENTER_INDEX, MAIN_LOBE_HALF_WIDTH, GUARD_BAND_HALF_WIDTH)

    return loss, pattern
    


def main():
    #v_model = np.random.randint(0,2095, SIZE)  #initially assume random voltages [0,10)
    #v_model = np.clip(np.round(INIT_VOLTAGE_MAP/DAC_MIN_STEP_SIZE), 0, 2047)
    voltages = np.round(np.random.uniform(2.0, 5.0, size=(12, 8)),3)

    print(voltages)

    nsi = NSI2000Client().connect()
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
    
    update_lb_array_file(INIT_VOLTAGE_MAP)
    rpi.update_dacs()
    time.sleep(40)
    print("Ready to run")
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("exiting")
            break
    nsi.disconnect()
    rpi.stop_program()
    rpi.close()
    
if __name__ == "__main__":
    main()