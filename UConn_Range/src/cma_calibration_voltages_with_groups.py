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

horn_inverse_old = np.array([
    [ 178.73,  165.82,  152.71,  148.15,  141.09,  143.24,  143.35,  153.25],
    [ -99.95, -120.14, -133.58, -145.87, -151.27, -155.93, -155.07, -146.10],
    [ -56.33,  -69.64,  -81.78,  -87.22,  -93.13,  -92.01,  -91.18,  -82.45],
    [  30.43,   10.03,   -2.08,  -14.96,  -19.72,  -24.70,  -24.63,  -15.12],
    [  79.27,   67.36,   54.82,   49.71,   43.41,   45.04,   45.29,   54.20],
    [ 170.96,  151.98,  139.70,  127.77,  121.86,  117.17,  118.39,  126.35],
    [-133.98, -145.98, -157.99, -163.49, -169.20, -168.56, -167.07, -159.14],
    [ -37.34,  -56.68,  -67.87,  -80.46,  -85.35,  -90.15,  -89.44,  -80.97],
    [ -56.68,  -67.87,  -80.46,  -85.35,  -91.57,  -90.15,  -89.44,  -80.97],
    [  41.68,   22.57,   11.85,   -0.71,   -5.68,  -10.27,   -9.52,   -0.96],
    [ 105.33,   93.70,   82.47,   76.51,   71.33,   71.91,   73.82,   80.75],
    [-152.45, -170.08,  178.01,  166.62,  160.92,  156.48,  158.37,  164.85]
])
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
# horn_inverse = np.clip(horn_inverse, -80, 80)

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
        params = coord_map(x) 
        voltages = np.round(voltage_from_phase(horn_inverse, params), 3)
        update_lb_array_file(voltages)
        #sends low and high band array files to PI and runs remote command to update DACs
        rpi.update_dacs()
        time.sleep(LC_DELAY_TIME)
        pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)
        return -pattern[0]  # maximize magnitude
    except Exception:
        return -1e9

def plotting(history, voltages_best_00, starting_sigmoid, best_params, restart_folder):
    iters = [h["iter"] for h in history]
    best_f = [h["best_f"] for h in history]
    sigmas = [h["sigma"] for h in history]

    plt.figure()
    plt.plot(iters, best_f, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value (fbest)")
    plt.title("CMA-ES Best Fitness Over Time")
    plt.grid(True)
    plt.savefig(restart_folder/ "FitnessVsiter.png", dpi=200)
    

    plt.figure()
    plt.plot(iters, voltages_best_00, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Voltage at element (V)")
    plt.title("Voltage Assigned to Single Element (Best Params)")
    plt.grid(True)
    plt.savefig(restart_folder/ "Voltageat00.png", dpi=200)

    plt.figure()
    plt.plot(iters, sigmas, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Sigma (step size)")
    plt.title("CMA-ES Step Size (σ) Over Time")
    plt.grid(True)
    plt.savefig(restart_folder/ "SigmaVsIter.png", dpi=200)
    
    plt.close('all')

    
sigma0 = 2         # explore ~20% of full scale at first

starting_params = np.array([1.34326172, 1.34326172, 0, 0, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 6.09594727,  6.09594727, 0, 0.57421875,  0.57421875, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 1.53808594,  1.53808594, 0.57421875,  0.57421875])
print(starting_params)

opts = {
    "popsize": 12,
    "maxfevals": 288,
    "verb_disp": 1,
    "bounds": [[0,0,0,0], [10,10,10,10]]
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



best_overall = None
restart_num = 0
best_f = float("inf")
num_iters = 3

for restart in range(num_iters):
    voltages_best_00 = []
    history = []
    restart_folder = exp_folder / f"Restart {restart}"
    restart_folder.mkdir(parents = True, exist_ok = False)
    es = cma.CMAEvolutionStrategy(starting_params, sigma0, opts)
    while not es.stop():
        X = es.ask()
        F = [objective(x, nsi, rpi) for x in X]
        es.tell(X, F)
        es.disp()
        history.append({
            "iter": es.countiter,
            "best_f": es.result.fbest,
            "best_x": np.array(es.result.xbest),
            "sigma": es.sigma
            })
        voltages_best_00.append(voltage_from_phase(horn_inverse, coord_map(np.array(es.result.xbest)))[0,0])
        plotting(history, voltages_best_00, starting_sigmoid, coord_map(np.array(es.result.xbest)), restart_folder)
        
    np.savez(restart_folder / f"cma_history_1_14_26_Restart{restart}.npz", history=np.array(history, dtype=object))
    
    if es.result.fbest < best_f:
        best_f = es.result.fbest
        restart_num = restart
        best_overall = np.array(es.result.xbest)
        
    starting_params = np.random.uniform(0, 10, size=4)
    starting_sigmoid = coord_map(starting_params)
    
best_params = coord_map(best_overall)     
print("Done")
print(f"Best parameters: {best_params}, Best fitness: {best_f}, During iter {restart_num+1} of {num_iters}")           # physical parameters

with open(exp_folder / f"Best Iteration was {restart_num+1} of {num_iters}.txt", "w") as f:
    f.write(f"Best Iteration was {restart_num+1} of {num_iters}.txt")

data = np.load(exp_folder / rf"Restart {restart_num}/cma_history_1_14_26_Restart{restart_num}.npz", allow_pickle=True)
history = data["history"].tolist()
best_x = np.array([h["best_x"] for h in history])
best_params = np.array([coord_map(x) for x in best_x])
v = np.linspace(0,10.5,500)
plt.figure()
for i, param in enumerate(best_params):
    if(i%2 == 0):
        continue
    plt.plot(v, phase_from_voltage(v, param), label=f"iter {i}")
plt.plot(v, phase_from_voltage(v, best_params[-1]), label="final", color="red", linewidth=2)
plt.xlabel("Voltage")
plt.ylabel("Phase")
plt.title("Phase vs voltage every sigmoid for best results")
plt.savefig(exp_folder/ f"EverysigmoidIter{restart_num}.png", dpi=200)
plt.legend()
plt.show()
plt.close()

zero_volts = np.zeros(SIZE)
update_lb_array_file(zero_volts)
rpi.update_dacs()
time.sleep(LC_DELAY_TIME)
    
nsi.disconnect()
rpi.stop_program()
rpi.close()
