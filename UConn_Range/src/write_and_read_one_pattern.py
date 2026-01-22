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

horn_inverse = np.array([
    [-39.439, -57.335, -85.701, -92.21 , -92.21 , -92.21 , -92.21 , -92.21 ],
    [-12.065, -39.439, -57.335, -69.278, -77.084, -77.084, -77.084, -57.335],
    [ 81.152,  28.798,  28.798, -12.065, -12.065, -12.065, -12.065, -12.065],
    [ 95.391,  81.152,  28.798,  28.798,  28.798,  28.798,  28.798,  28.798],
    [-92.21 ,  95.759,  95.759,  95.759,  95.759,  95.759,  95.759,  95.759],
    [-92.21 , -92.21 , -92.21 ,  95.759,  95.759,  95.759,  95.759,  95.759],
    [-39.439, -77.084, -90.452, -92.21 , -92.21 , -92.21 , -92.21 , -92.21 ],
    [-12.065, -12.065, -39.439, -39.439, -57.335, -57.335, -39.439, -39.439],
    [ 95.759,  81.152,  81.152,  28.798,  28.798,  28.798,  28.798,  28.798],
    [ 95.759,  95.759,  95.759,  95.759,  95.759,  95.759,  95.759,  95.759],
    [-89.46 , -92.21 , -92.21 , -92.21 , -92.21 , -92.21 , -92.21 , -92.21 ],
    [-12.065, -39.439, -57.335, -69.278, -82.294, -82.294, -82.294, -69.278]
])
'''
INIT_VOLTAGE_MAP = np.array([
    [ 3.0007,  3.4983,  5.4988, 10.0024, 10.0024, 10.0024, 10.0024, 10.0024],
    [ 2.4980,  3.0007,  3.4983,  4.0010,  4.4985,  4.4985,  4.4985,  3.4983],
    [ 1.4978,  2.0005,  2.0005,  2.4980,  2.4980,  2.4980,  2.4980,  2.4980],
    [ 1.0002,  1.4978,  2.0005,  2.0005,  2.0005,  2.0005,  2.0005,  2.0005],
    [10.0024,  0.4976,  0.4976,  0.4976,  0.4976,  0.4976,  0.4976,  0.4976],
    [10.0024, 10.0024, 10.0024,  0.4976,  0.4976,  0.4976,  0.4976,  0.4976],
    [ 3.0007,  4.4985,  7.0017, 10.0024, 10.0024, 10.0024, 10.0024, 10.0024],
    [ 2.4980,  2.4980,  3.0007,  3.0007,  3.4983,  3.4983,  3.0007,  3.0007],
    [ 0.4976,  1.4978,  1.4978,  2.0005,  2.0005,  2.0005,  2.0005,  2.0005],
    [ 0.4976,  0.4976,  0.4976,  0.4976,  0.4976,  0.4976,  0.4976,  0.4976],
    [ 6.4990, 10.0024, 10.0024, 10.0024, 10.0024, 10.0024, 10.0024, 10.0024],
    [ 2.4980,  3.0007,  3.4983,  4.0010,  5.0012,  5.0012,  5.0012,  4.0010]
])
'''
INIT_VOLTAGE_MAP = np.array([
    [ 6.067 ,  0.6278,  0.8823,  0.2486,  2.006 ,  5.4271,  0.321 ,  0.0469],
    [ 0.7559,  7.3297,  2.6697,  0.0045,  2.0867,  0.2098,  0.2118,  0.773 ],
    [ 9.6589, 10.3589, 10.4359,  8.4674, 10.4822,  9.487 ,  9.8624, 10.47  ],
    [10.3727,  8.2664,  7.6164,  9.3685, 10.3965, 10.0805,  9.1316,  8.2544],
    [ 9.386 , 10.1932, 10.4655,  6.3775,  7.0003,  7.9004,  9.0392, 10.0604],
    [ 8.66  ,  5.7372,  7.4983,  7.3074,  9.4803,  0.3977,  5.7029,  4.7777],
    [ 1.5046,  0.0268,  0.0906,  0.0985,  5.0471,  0.5312,  1.1052,  0.3222],
    [ 1.3236,  0.5411,  3.0841,  4.029 ,  0.3869,  0.8865,  0.24  ,  0.794 ],
    [ 9.1537,  7.6093,  8.6891,  9.2877, 10.0263, 10.4867,  8.5141,  8.6959],
    [ 7.5481, 10.4933,  7.3484,  9.8683, 10.4946,  9.9176,  9.488 ,  8.4838],
    [ 1.2169,  1.6436,  1.8677,  0.9525,  3.1095,  1.9384,  0.5165,  1.9519],
    [ 0.7467,  1.9374,  1.2276,  1.7579,  0.8989,  1.1013,  0.5472,  0.1683],
])
  
#Size of array representing elements on the board-12x8 for LB
SIZE = (12,8)

LC_DELAY_TIME = 40 #in secs

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

#Set the beam (corresponds to frequency measured) number that you put in NSI software. 19.3 Ghz is ideal for low band
BEAM = 1

ELEVATION = 0
AZIMUTH = 0

FREQUENCY = "19.7 Ghz"

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
    
def voltage_from_phase(phi, params, eps=1e-4):
    A, k, v0, y0 = params

    q = (phi - y0) / A
    q = np.clip(q, eps, 1 - eps)

    V = v0 + (1.0 / k) * np.log(q / (1 - q))
    
    return np.clip(V, 0.0, 10.5)

def main():
    #v_model = np.random.randint(0,2095, SIZE)  #initially assume random voltages [0,10)
    #v_model = np.clip(np.round(INIT_VOLTAGE_MAP/DAC_MIN_STEP_SIZE), 0, 2047)
    #v_base = np.round(np.random.uniform(2.0, 5.0, 12),3)
    #voltages = np.tile(v_base[:,None], (1,8))
    #params = np.array([1.51268964e+0,-2.76195609e-02,9.49768775e-01,-6.45534204e+01])
    #voltages = voltage_from_phase(horn_inverse, params)
    #voltages = INIT_VOLTAGE_MAP
    #data = np.load(r"C:\Users\NSI-MI\Downloads\grid.npz")
    #voltages = data["voltages"]
    #index = data["index_arr"]
    voltages = np.zeros((12,8))
    voltages[11,7] = 10
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

    
    update_lb_array_file(voltages)
    rpi.update_dacs()
    pattern = nsi.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)
    print(pattern)
   
        
    
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
    time.sleep(40)
    
    nsi.disconnect()
    rpi.stop_program()
    rpi.close()

if __name__ == "__main__":
    main()
