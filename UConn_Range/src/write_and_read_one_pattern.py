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
PI_HOST = "RasPI5.local"
PI_HOST = "10.194.115.111"
USERNAME = "feix"         
PASSWORD = "password"          
KEY_FILE = None # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"
PI_PORT = 22

STOP_FILE = r"/home/feix/STOP.txt"
LOCAL_FILE_HB = r"C:\NSI2000\Data\Carillon\HB_voltages.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = r"C:\NSI2000\Data\Carillon\LB_voltages.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = r"/home/feix/Desktop/dataHB.csv"  # where to put it on the Pi
REMOTE_FILE_LB = r"/home/feix/Downloads/2025-12-18 VoltageMap_HornCorrection.csv"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/feix/Gen3DAC60096EVM_SPI_RPi5_schollV2.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"nohup python3 {REMOTE_PROGRAM} >/dev/null 2>&1 &"


INIT_VOLTAGE_MAP = np.array([
    [0.0, 0.43082805, 0.36113632, 0.0, 0.0, 0.36113632, 0.43082805, 0.0],
    [0.0, 0.40098652, 1.01827158, 0.0, 0.0, 1.01827158, 0.40098652, 0.0],
    [10.49487305, 7.50081875, 3.33448287, 10.49487305, 10.49487305, 3.33448287, 7.50081875, 10.49487305],
    [10.49487305, 9.12302114, 8.7452768, 10.49487305, 10.49487305, 8.7452768, 9.12302114, 10.49487305],
    [10.49487305, 6.06508177, 9.73930833, 10.49487305, 10.49487305, 9.73930833, 6.06508177, 10.49487305],
    [2.40454102, 1.6955036, 4.43498863, 2.40454102, 2.40454102, 4.43498863, 1.6955036, 2.40454102],
    [0.0, 0.89978374, 0.61801542, 0.0, 0.0, 0.61801542, 0.89978374, 0.0],
    [0.38452148, 0.50618363, 0.29721435, 0.38452148, 0.38452148, 0.29721435, 0.50618363, 0.38452148],
    [10.49487305, 8.2179479, 10.44846116, 10.49487305, 10.49487305, 10.44846116, 8.2179479, 10.49487305],
    [2.40454102, 6.96661354, 10.24318273, 2.40454102, 2.40454102, 10.24318273, 6.96661354, 2.40454102],
    [1.75341797, 5.3269361, 1.50438538, 1.75341797, 1.75341797, 1.50438538, 5.3269361, 1.75341797],
    [0.57421875, 0.75419425, 0.8595833, 0.57421875, 0.57421875, 0.8595833, 0.75419425, 0.57421875]
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
    data2=np.load(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\Experiments\column_search\Calibration_2026-02-19_14-02-25\run_2\finalvoltagesandlosses.npz")

    

    
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

    
    voltages = np.array([
    [1.11103, 2.07747, 4.44963, 10.5, 0, 1.52362, 3.55497, 10.5],
    [1.41815, 3.20992, 10.5, 0, 1.30698, 2.75038, 10.5, 0],
    [0.31626, 2.24564, 8.40103, 0, 1.02212, 2.23965, 6.38328, 0],
    [1.4577, 4.81089, 10.5, 0.59264, 1.81207, 3.88447, 10.5, 0.42402],
    [0.36301, 2.6917, 10.5, 0, 1.37982, 2.62605, 8.24972, 0],
    [1.71069, 6.67968, 0, 0.84195, 1.93361, 4.19646, 10.5, 0.74523],
    [0.77203, 3.12103, 10.5, 0, 1.3459, 2.50597, 7.58607, 0],
    [1.87234, 6.59449, 0, 0.59199, 1.70732, 3.37595, 10.5, 0.47303],
    [0.96714, 2.81839, 10.5, 0, 0.95532, 2.04683, 5.10142, 10.5],
    [1.61929, 3.78821, 10.5, 0, 1.22133, 2.43451, 7.43496, 0],
    [0.55848, 1.83685, 4.19987, 10.5, 0, 1.47013, 2.8581, 8.53369],
    [0.55096, 1.74073, 4.0709, 10.5, 0.34624, 1.69141, 2.94388, 5.32943]
])
    voltages_hb_00  = np.array([
    [2.10903, 6.30077, 0, 2.22169, 10.4, 0, 2.14208, 2.52367],
    [2.2028, 10.4, 0, 2.37654, 10.4, 0.19684, 2.29267, 4.19474],
    [1.85079, 2.86224, 0, 2.07013, 10.4, 0, 2.11234, 2.86025],
    [1.93102, 4.4346, 0, 2.1999, 10.4, 0, 2.22623, 5.61829],
    [2.35168, 10.4, 1.72448, 3.28527, 0, 1.96068, 2.67997, 10.4],
    [2.4161, 10.4, 1.9557, 5.52256, 0, 2.06712, 3.50495, 10.4],
    [2.0034, 10.4, 0, 2.40702, 10.4, 0.83411, 2.40399, 10.4],
    [2.01447, 10.4, 0, 2.50952, 10.4, 1.53378, 2.4772, 10.4],
    [2.52036, 0, 2.1311, 10.4, 0, 2.17344, 7.64909, 0],
    [2.54954, 0, 2.17395, 10.4, 0, 2.19295, 10.4, 0],
    [1.99914, 10.4, 1.0043, 2.70702, 10.4, 1.75036, 2.5523, 10.4],
    [1.97302, 10.4, 1.21845, 2.7433, 10.4, 1.71742, 2.53189, 10.4],
    [2.53621, 0, 2.19788, 10.4, 0, 2.17128, 10.4, 0],
    [2.49757, 0, 2.17907, 10.4, 0, 2.12898, 5.93706, 0],
    [1.87092, 10.4, 0.62969, 2.59695, 10.4, 0.95202, 2.40442, 10.4],
    [1.77341, 10.4, 0, 2.48791, 10.4, 0, 2.3202, 10.4],
    [2.38786, 0, 2.07103, 6.95936, 0, 1.94901, 2.84508, 10.4],
    [2.3247, 10.4, 1.95973, 3.83344, 0, 1.69226, 2.51211, 10.4],
    [1.04303, 6.70909, 0, 2.27286, 10.4, 0, 2.10984, 5.5035],
    [0.01587, 3.84296, 0, 2.15123, 10.4, 0, 1.93431, 2.78341],
    [2.18136, 10.4, 0, 2.40318, 10.4, 0, 2.18707, 10.4],
    [2.09191, 10.4, 0, 2.23194, 10.4, 0, 1.99334, 2.89095],
    [0, 2.44075, 10.4, 1.35563, 2.47518, 10.4, 0, 2.19363],
    [0, 2.28412, 10.4, 0, 2.25074, 10.4, 0, 1.92702]
])
    #voltages=np.full((24,8), 10)
    update_lb_array_file(voltages, dual_band = False)
    #update_hb_array_file(voltages_hb_00, dual_band = True)
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
    v_close = np.round(np.zeros((24,8)),3)
    update_hb_array_file(v_close)
    rpi.update_dacs()
    #time.sleep(40)
    #nsi.save_scan(r"C:\NSI2000\Data\Carillon\temp.asc")
    nsi.disconnect()
    rpi.stop_program()
    rpi.close()

if __name__ == "__main__":
    main()
