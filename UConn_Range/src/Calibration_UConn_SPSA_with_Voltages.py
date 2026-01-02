# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:17 2025

@author: SchollJamesAC3CARILL
96 Element LB reflectarray calibration

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
     [1.859, 1.859, 1.859, 1.928, 1.995, 1.995, 1.948, 1.889],
     [9.769, 9.769, 9.769, 1.859, 1.859, 1.859, 1.859, 1.859],
     [9.769, 9.769, 9.769, 9.769, 9.769, 9.769, 9.769, 9.769],
     [2.577, 2.784, 3.104, 3.767, 4.87,  5.109, 5.109, 4.065],
     [2.352, 2.417, 2.472, 2.504, 2.529, 2.535, 2.509, 2.485],
     [1.948, 2.117, 2.188, 2.248, 2.276, 2.276, 2.28,  2.256],
     [1.859, 1.859, 1.859, 1.859, 1.859, 1.859, 1.859, 1.859],
     [9.769, 9.769, 9.769, 9.769, 9.769, 9.769, 9.769, 9.769],
     [2.893, 3.539, 5.782, 9.769, 9.769, 9.769, 9.769, 8.444],
     [2.34,  2.406, 2.451, 2.504, 2.535, 2.54,  2.546, 2.514],
     [1.976, 2.117, 2.188, 2.217, 2.243, 2.248, 2.226, 2.203],
     [9.769, 9.769, 1.859, 1.859, 1.859, 1.859, 1.859, 1.859],
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
    V = np.round(V * DAC_MIN_STEP_SIZE, 3)
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
    
def calibration_step(v, k, vna_instance, rpi, cal_folder):
    """
    Perform one SPSA iteration updating v.
    Returns updated v and (L_plus, L_minus).
    """
    ak = max(int(a0 / (k + 1) ** alpha), 1)  # learning rate
    ck = max(int(c0 / (k + 1) ** gamma), 1)  # perturbation magnitude
    
    # random ±1 perturbations per element
    delta_v = np.random.choice([-1.0, 1.0], size=SIZE)

    # plus / minus parameter sets
    v_plus = v + ck * delta_v
    v_minus = v - ck * delta_v
    
    v_minus = np.clip(v_minus, 0, 2047)
    v_plus = np.clip(v_plus, 0, 2047)

    # evaluate loss for each perturbed set
    L_plus, pattern_plus = compute_loss(v_plus, vna_instance, rpi, k, True, cal_folder)
    L_minus, pattern_minus = compute_loss(v_minus, vna_instance, rpi, k, False, cal_folder)

    # scalar difference
    diff = L_plus - L_minus

    # SPSA gradient estimate for each parameter:
    #   g_k = (L+ - L-) / (2 ck Δ_k)
    g_v = diff / (2.0 * ck) * (1.0 / delta_v)

    # gradient descent update: param <- param - ak * g
    v_new = v - ak * g_v

    v_new = np.clip(v_new, 0, 2047)
    
    return v_new, L_plus, L_minus, pattern_plus, pattern_minus, v_plus, v_minus, ak, ck
    

def main():
    #v_model = np.random.randint(0,2095, SIZE)  #initially assume random voltages [0,10)
    v_model = np.clip(np.round(INIT_VOLTAGE_MAP/DAC_MIN_STEP_SIZE), 0, 2047)
    lp_arr = []
    pattern_point_arr = []
    all_patternsp = []
    all_patternsm = []
    all_voltages = [] #All model voltages
    all_voltages_pp = []#all plus perterbed voltages 
    all_voltages_pm = []#all minus perturbed voltages
    ak_arr = []
    ck_arr = []
    v_arr = []
    v_diff_arr = []
    diff_arr = []
    lavg_arr = []
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
    
    experiment_dir = Path(EXP_DIR)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder = experiment_dir / f"Calibration_{ts}"
    raw_folder = experiment_dir / exp_folder / "raw"
    exp_folder.mkdir(parents = True, exist_ok = False)
    raw_folder.mkdir(parents = True, exist_ok = False)
    
    print("Starting SPSA calibration...")
    t0 = time.time()
    all_voltages.append(v_model)
    for k in range(num_iters):
        print(f"Iter: {k+1}")
        v_old = v_model[0][0]
        all_voltages.append(v_model) #gets all input voltages except the last
        v_model, Lp, Lm, patternp, patternm, vp, vm, ak, ck = calibration_step(v_model, k, nsi, rpi, raw_folder)
        lp_arr.append(Lp)
        pattern_point_arr.append(patternp[18])
        all_patternsp.append(patternp)
        all_patternsm.append(patternm)
        all_voltages_pp.append(vp)
        all_voltages_pm.append(vm)
        ak_arr.append(ak)
        ck_arr.append(ck)
        v_arr.append(v_model[0][0])
        v_diff_arr.append(abs(v_model[0][0]-v_old))
        diff_arr.append(abs(Lp-Lm))
        lavg_arr.append(abs(Lp+Lm/2))
        
        t1 = time.time()

        if k+1 % 5 == 0 or k == num_iters - 1:
            # Also compute loss at current (unperturbed) parameters for logging
            Lp_print = float(Lp) #convert to python scalar for printing
            Lm_print = float(Lm)
            v_model_print=float(v_model[0][0])
            print(
                f"Iter {k+1:03d} | L+={Lp_print:.4f} L-={Lm_print:.4f} "
                f"v[0][0]={v_model_print:.1f} dt={t1-t0:.3f}s"
                )
    diff_arr_np = np.array(diff_arr)
    mean = np.mean(diff_arr_np)
    print(f"Loss difference mean: {mean}")
    
    nsi.disconnect()
    rpi.stop_program()
    rpi.close()
    
    params = {
            "algorithm": "SPSA perturbing voltages",
            "a0": a0,
            "c0": c0,
            "alpha": alpha,
            "gamma": gamma,
            "num_iters": num_iters,
            "pi_program_ran": REMOTE_PROGRAM,
            "low_or_high_band": "Low Band",
            "frequency": FREQUENCY,
            "main_lobe_half_width": MAIN_LOBE_HALF_WIDTH,
            "center_index": CENTER_INDEX,
            "guard_band_half_width": GUARD_BAND_HALF_WIDTH,
            "loss_equation": "loss = 1 - E_main / (E_main + E_side)",
            "notes": "Ignores sidelobes close to transmitter, +x dir"  
        }
    
    with open(exp_folder / "params.json", "w") as f:
        json.dump(params, f, indent=2)
        
    np.savez(
        exp_folder / "results.npz",
        final_voltages= v_model,
        all_voltages= all_voltages,
        all_patternsp= all_patternsp,
        all_patternsm= all_patternsm,
        all_voltages_pp= all_voltages_pp,
        all_voltages_pm= all_voltages_pm,
        init_voltage_map=INIT_VOLTAGE_MAP
    )
    
    plot_dir = exp_folder / "plots"
    plot_dir.mkdir()
    
    #Plot Loss
    fig1, ax1 = plt.subplots()
    x = np.arange(1,len(lp_arr)+1)
    ax1.plot(x,lp_arr)
    ax1.set_title("Loss_plus")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Lp")
    ax1.grid()
    fig1.savefig(plot_dir / "LpVsIter.png", dpi=200)

    
    #Plot perterbattion scale
    fig2, ax2 = plt.subplots()
    ax2.plot(x,ck_arr)
    ax2.set_title("c_k")
    ax2.set_xlabel("Iteration (k)")
    ax2.set_ylabel("c_k")
    ax2.grid()
    fig2.savefig(plot_dir / "c_kVsIter.png", dpi=200)
    
    #Plot learning scale
    fig3, ax3 = plt.subplots()
    ax3.plot(x,ak_arr)
    ax3.set_title("a_k")
    ax3.set_xlabel("Iteration (k)")
    ax3.set_ylabel("a_k")
    ax3.grid()
    fig3.savefig(plot_dir / "a_kVsIter.png", dpi=200)
    
    #Plot magnitude received in center
    fig4, ax4 = plt.subplots()
    ax4.plot(x,pattern_point_arr)
    ax4.set_title(f"Magnitude at {FREQUENCY} in Center for Loss_plus")
    ax4.set_xlabel("Iteration (k)")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.grid()
    fig4.savefig(plot_dir / "MagInCenterVsIter.png", dpi=200)
    
    #Plot magnitude vs span for every iteration step
    fig5, ax5 = plt.subplots()
    step = 1
    run_idx = np.arange(0, num_iters, step)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=run_idx[0], vmax=run_idx[-1])
    for i in run_idx:
        if i == num_iters - 1:
            continue
        ax5.plot(all_patternsp[i], color=cmap(norm(i)), alpha=0.7)
    ax5.plot(all_patternsp[-1], color="red", linewidth=2, label="Final run")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig5.colorbar(sm, ax=ax5)
    cbar.set_label("Run index")
    cbar.set_ticks(run_idx)
    cbar.set_ticklabels(run_idx)
    
    ax5.set_title(f"Magnitude vs span every {step} run")
    ax5.set_xlabel("span: -2.5 to 2.5 in")
    ax5.set_ylabel("Magnitdue (dB)")
    ax5.grid()
    fig5.savefig(plot_dir / f"MagVsSpanevery{step}run.png", dpi=200)
    
    #Plot voltage at at element 00
    fig6, ax6 = plt.subplots()
    v_arr = np.array(v_arr, dtype=float)
    ax6.plot(x, np.round(v_arr*DAC_MIN_STEP_SIZE,3))
    ax6.set_title("Voltage at element [0][0]")
    ax6.set_xlabel("Iteration (k)")
    ax6.set_ylabel("Voltage (V)")
    ax6.grid()
    fig6.savefig(plot_dir / "VoltageAt00VsIter.png", dpi=200)
    
    fig7, ax7 = plt.subplots()
    ax7.plot(x, diff_arr)
    ax7.set_title("|Lp-Lm| vs Iteration")
    ax7.set_xlabel("Iteration (k)")
    ax7.set_ylabel("|Lp-Lm|")
    ax7.grid()
    fig7.savefig(plot_dir / "LpminusLmVsIter.png", dpi=200)
    
    fig8, ax8 = plt.subplots()
    v_diff_arr = np.array(v_diff_arr, dtype=float)
    ax8.plot(x, v_diff_arr*DAC_MIN_STEP_SIZE)
    ax8.set_title("|Vnew-Vold| vs Iteration at element [0][0]")
    ax8.set_xlabel("Iteration (k)")
    ax8.set_ylabel("V")
    ax8.grid()
    fig8.savefig(plot_dir / "VoltageDiffAt00VsIter.png", dpi=200)
    
    fig9, ax9 = plt.subplots()
    ax9.plot(x, lavg_arr)
    ax9.set_title("Lp+Lm/2 vs Iteration")
    ax9.set_xlabel("Iteration (k)")
    ax9.set_ylabel("Loss")
    ax9.grid()
    fig9.savefig(plot_dir / "LossAvgVsIter.png", dpi=200)

    plt.show()
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    plt.close(fig7)
    plt.close(fig8)
    plt.close(fig9)
    print("Done.")
    
if __name__ == "__main__":
    main()