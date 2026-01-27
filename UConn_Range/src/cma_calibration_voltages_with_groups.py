# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:58:32 2026

@author: SchollJamesAC3CARILL
"""

import numpy as np
import cma
from scipy.optimize import curve_fit
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

LC_DELAY_TIME =40 #in secs

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

#Set the beam (corresponds to frequency measured) number that you put in NSI software. 19.3 Ghz is ideal for low band
BEAM = 1
SIZE = (12, 8)

#Loss function params
MAIN_LOBE_HALF_WIDTH = 2 #Number of points in scan for the main lobe half width
CENTER_INDEX = 71 #Index where the center lobe should be
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

def objective(x, vna_instance, rpi):
    try:
        voltages = np.tile(x[:,None], (1,8))
        update_lb_array_file(voltages)
        #sends low and high band array files to PI and runs remote command to update DACs
        rpi.update_dacs()
        time.sleep(LC_DELAY_TIME)
        pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)
        loss = loss_center_vs_sidelobes_db(pattern, CENTER_INDEX, MAIN_LOBE_HALF_WIDTH, GUARD_BAND_HALF_WIDTH)
        return loss, pattern  # maximize magnitude
    except Exception:
        return 1e9
    
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
    side_mask[g0:g1] = False  # everything outside guard is "sidelobes"
    #side_mask[g0:] = False  # everything outside guard is including all positive indices which are closest to horn (from milad- it interferes)
    
    E_main = float(np.sum(pwr[main_mask]))
    E_side = float(np.sum(pwr[side_mask]))

    loss = 1 - E_main / (E_main + E_side)
    
    #lm = .25
    #loss = -E_main - lm * E_side
    
    return loss

def plotting(history, voltages_best_00, exp_folder):
    iters = [h["iter"] for h in history]
    best_f = [h["best_f"] for h in history]
    sigmas = [h["sigma"] for h in history]

    plt.figure()
    plt.plot(iters, best_f, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value (fbest)")
    plt.title("CMA-ES Best Fitness Over Time")
    plt.grid(True)
    plt.savefig(exp_folder/ "FitnessVsiter.png", dpi=200)
    

    plt.figure()
    plt.plot(iters, voltages_best_00, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Voltage at element (V)")
    plt.title("Voltage Assigned to Single Element (Best Params)")
    plt.grid(True)
    plt.savefig(exp_folder/ "Voltageat00.png", dpi=200)

    plt.figure()
    plt.plot(iters, sigmas, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Sigma (step size)")
    plt.title("CMA-ES Step Size (σ) Over Time")
    plt.grid(True)
    plt.savefig(exp_folder/ "SigmaVsIter.png", dpi=200)
    
    plt.close('all')

    
sigma0 = 2    # explore ~20% of full scale at first

#starting_params = starting_params.flatten()
starting_params = np.round(np.random.uniform(0, 10.49487305, 12),3)

opts = {
    "popsize": 20,
    "maxfevals": 400,
    "verb_disp": 1,
    "bounds": [[0]*12, [10.49487305]*12]
}

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


experiment_dir = Path(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\Experiments\CMA-ES")
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_folder = experiment_dir / f"Calibration_{ts}"
exp_folder.mkdir(parents = True, exist_ok = False)

curr_best = 1
voltages_best_00 = []
history = []
pattern_arr = []
es = cma.CMAEvolutionStrategy(starting_params, sigma0, opts)
while not es.stop():
    patterns = []
    X = es.ask()
    F, patterns = zip(*(objective(x, nsi, rpi) for x in X))
    es.tell(X, F)
    es.disp()
    
    if curr_best != es.result.fbest:
        F = np.array(F)
        curr_best = es.result.fbest
        idx = np.argmax(F==es.result.fbest)
        pattern_arr.append(patterns[idx])
        plt.figure()
        for i, p in enumerate(pattern_arr):
            plt.plot(p, label=f"Pattern {i}") 
        plt.xlabel("Span -10 to +10in")
        plt.ylabel("Mag (dB)")
        plt.title(f"Mag vs Span best")
        plt.grid(True)
        plt.legend()
        plt.savefig(exp_folder/ f"MagVsSpanBest.png", dpi=200)
        plt.show()
        
    history.append({
        "iter": es.countiter,
        "best_f": es.result.fbest,
        "best_x": np.array(es.result.xbest),
        "sigma": es.sigma
        })
    voltages_best_00.append(es.result.xbest[0])
    plotting(history, voltages_best_00, exp_folder)
    
history.append({"best_patterns": np.array(pattern_arr)})
np.savez(exp_folder / "cma_history_1_21_26.npz", history=np.array(history, dtype=object))
    
       
print("Done")
print(f"Best parameters: {es.result.xbest}, Best fitness: {es.result.fbest}")



zero_volts = np.zeros(SIZE)
update_lb_array_file(zero_volts)
rpi.update_dacs()
time.sleep(LC_DELAY_TIME)
nsi.disconnect()
rpi.stop_program()
rpi.close()
