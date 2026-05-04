# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:23:27 2026

@author: SchollJamesAC3CARILL
"""

import numpy as np
import matplotlib.pyplot as plt
from Shared.PiController import PiController
from Shared.NSI2000Client import NSI2000Client
import time
from datetime import datetime
from pathlib import Path

#Place to store experiment results
EXP_DIR = r"C:\NSI2000\Data\Carillon\reflectarray_calibration\Experiments"

SCAN_FILENAME = r"C:\NSI2000\Data\Carillon\calibration_scan_real.nsi"

#PHASE_MAP_FILE_LB = r"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\phases_with_beam_steering_0theta_0phi_hex_12x8.txt" 

PI_HOST = "10.194.115.111"#IP of PI controlling DACs
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
REMOTE_COMMAND = f"python3 {REMOTE_PROGRAM}"

LC_DELAY_TIME =1 #in secs

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

#Set the beam (corresponds to frequency measured) number that you put in NSI software. 19.3 Ghz is ideal for low band
BEAM = 1
SIZE = (12, 8)

#Loss function params
MAIN_LOBE_HALF_WIDTH = 2 #Number of points in scan for the main lobe half width
CENTER_INDEX = 50 #Index where the center lobe should be
GUARD_BAND_HALF_WIDTH = 4 #Number of points in the scan for a guard band not considered in loss function

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
    main_half_width: int,
    guard_half_width: int,
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
    side_mask[g0:g1] = False  # everything outside guard is "sidelobes"
    #side_mask[g0:] = False  # everything outside guard is including all positive indices which are closest to horn (from milad- it interferes)
    
    E_main = float(np.sum(pwr[main_mask]))
    E_side = float(np.sum(pwr[side_mask]))

    loss = 1 - E_main / (E_main + E_side)
    
    #lm = .25
    #loss = -E_main - lm * E_side
    
    #E_main = np.sum(pwr)
    return loss

def compute_loss(v, vna_instance, rpi):
    #updates low band array file on local computer
    update_lb_array_file(v)
    
    #sends low and high band array files to PI and runs remote command to update DACs
    rpi.update_dacs()
    
    time.sleep(LC_DELAY_TIME)
      
    pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)[:,0]
    print(pattern)
    loss = loss_center_vs_sidelobes_db(pattern, CENTER_INDEX, MAIN_LOBE_HALF_WIDTH, GUARD_BAND_HALF_WIDTH)
    return loss, pattern        
            
        
possible_voltages = np.arange(0, 10.5, DAC_MIN_STEP_SIZE)
idx_low = 300 #roughly 2V
idx_high = 975 #roughly 5V

# choose how many points per region
n_low = 8     # 0–2 V
n_mid = 16    # 2–5 V (denser)
n_high = 6    # 5–10.5 V

v_low  = np.round(np.linspace(0, idx_low, n_low, endpoint=False)).astype(int)
v_mid  = np.round(np.linspace(idx_low, idx_high, n_mid, endpoint=False)).astype(int)
v_high = np.round(np.linspace(idx_high, 2047, n_high)).astype(int)

voltages = np.concatenate([possible_voltages[v_low], possible_voltages[v_mid], possible_voltages[v_high]])

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

experiment_dir = Path(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\Experiments\column_search")
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_folder = experiment_dir / f"Calibration_{ts}"
exp_folder.mkdir(parents = True, exist_ok = False)

final_voltages = np.zeros((12,8))
#v_arr = np.zeros((12,8))
v_arr = np.array([
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
    [1.53808594, 1.15356445, 1.34326172, 1.75341797, 9.39770508, 1.15356445, 1.34326172, 1.75341797],
], dtype=np.float64)
for run in range(2):
    rand_cols = np.random.permutation(8)
    final_losses = []
    final_patterns = []
    run_folder = exp_folder / f"run_{run}"
    run_folder.mkdir(parents = True, exist_ok = False)
    for i in rand_cols:
        loss_arr = []
        pattern_arr = []
        for val in voltages:
            v_arr[:, i] = val
            loss, pattern = compute_loss(v_arr, nsi, rpi)
            loss_arr.append(loss)
            pattern_arr.append(pattern)
        loss_arr = np.array(loss_arr)
        min_val = loss_arr.min()
        min_index = loss_arr.argmin()
        final_voltages[:, i] = voltages[min_index]
        final_losses.append(min_val)
        final_patterns.append(pattern_arr[min_index])
        v_arr[:, i] = voltages[min_index]
        print(f"Min loss for col {i}: {min_val} at voltage {voltages[min_index]}V")
        plt.figure()
        plt.plot(voltages, loss_arr)
        plt.xlabel("Voltages")
        plt.ylabel("Loss")
        plt.title(f"Loss vs applied voltage col {i}")
        plt.grid(True)
        plt.savefig(run_folder/ f"LossvsAppliedVoltageCol{i}.png", dpi=200)
        
        plt.figure()
        for j, pattern in enumerate(pattern_arr):
            if j%2 != 0: #plot only even
                continue
            plt.plot(pattern, label=f"voltage {j}")
        plt.xlabel("y (-20 to 20in)")
        plt.ylabel("Mag (dB)")
        plt.title("Mag vs y each voltage")
        plt.grid(True)
        plt.legend()
        plt.savefig(run_folder/ f"MagVsSpanIter{i}.png", dpi=200)
        
        plt.show()
        plt.close("all")
        
    print(final_voltages)
    print(final_losses)
    plt.figure()
    for i, pattern in enumerate(final_patterns):
        if i%2 != 0: #plot only even
            continue
        plt.plot(pattern, label=f"Iter {i}")
    plt.xlabel("y (-20 to 20in)")
    plt.ylabel("Mag (dB)")
    plt.title("Mag vs Span after every col")
    plt.grid(True)
    plt.legend()
    plt.savefig(run_folder/ "MagVsSpanAfterEveryCol.png", dpi=200)

    plt.show()
    plt.close("all")

    np.savez(run_folder / "finalvoltagesandlosses.npz", 
             final_voltages=np.array(final_voltages),
             final_losses=np.array(final_losses),
             final_patterns=np.array(final_patterns)
             )
    #start with the previous run voltages
    v_arr = final_voltages


zero_volts = np.zeros(SIZE)
update_lb_array_file(zero_volts)
rpi.update_dacs()
time.sleep(LC_DELAY_TIME)
    
nsi.disconnect()
rpi.stop_program()
rpi.close()
