# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:17 2025
Sends 8x12 HB voltages at 27.2Ghz to 0-100V array
@author: SchollJamesAC3CARILL
Get latest for first time with git clone https://github.com/james-scholl-ct/Array-Calibration.git then use git pull
Install packages from Array-Calibration folder with pip install -e .
"""
import numpy as np
import math
import time
from typing import Optional
import matplotlib.pyplot as plt
from shared.PiController import PiController
from shared.VnaInstance import VnaInstance


PHASE_MAP_FILE_LB = rf"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\phases_with_beam_steering_0theta_0phi_hex_12x8.txt" 

PI_HOST = "192.168.6.100" #IP of PI controlling DACs
USERNAME = "ldantes"         
PASSWORD = "password"          
KEY_FILE = None               # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"

PI_PORT = 22
STOP_FILE = "/home/ldantes/ReflecTek_Pi/STOP.txt"
LOCAL_FILE_HB = rf"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\HB.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = rf"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\LB.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = "/home/ldantes/Desktop/data.csv"  # where to put it on the Pi
REMOTE_FILE_LB = "/home/ldantes/Desktop/LB.txt"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/ldantes/ReflecTek_Pi/spi_scholl.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"python3 {REMOTE_PROGRAM}"

IP_ADDR_VNA = "TCPIP0::192.168.6.150::inst0::INSTR" #VNA IP addr
#VNA sweep params
START = 27.2e9
STOP = 27.3e9
POINTS = 2

SIZE = (8,12)

DAC_MIN_STEP_SIZE = 200/16384

# SPSA hyperparameters
a0 = 3000  # learning-rate scale in dac steps
c0 = 6000  # perturbation scale in DAC steps should be 2-5x a0
alpha = 0.6 #.6-.8
gamma = 0.1
num_iters = 1

def read_phase_map_file(filename):
    phasemap = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            phasemap.append([int(x) for x in line.split(",")])
    phasemap = np.flipud((np.array(phasemap).T))#transpose then flip up/down
    return phasemap

def update_hb_array_file(V):
    V = np.clip(np.round(V * DAC_MIN_STEP_SIZE, 4), 0, 100)
    with open(LOCAL_FILE_HB, "w") as f:
        for row in V:
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")
    #This program is for HB only
    #with open(LOCAL_FILE_LB, "w") as f:
    #   for row in np.zeros((24,8)):
    #       line = ",".join(str(x) for x in row)
    #       f.write(line + "\n")


def get_pattern(vna_instance):
    pattern = vna_instance.sweep(START, STOP, POINTS)
    return pattern


def compute_loss(v, vna_instance, rpi):
    """
    This is what you'd do with real hardware:
      1. send V to array
      2. measure pattern
      3. compute scalar loss

    Here we simulate the pattern.
    """
    #updates high band array file on local computer
    update_hb_array_file(v)
    #sends low and high band array files to PI and runs remote command to update DACs
    rpi.update_dacs()
    time.sleep(1)
    pattern = get_pattern(vna_instance)
    magnitude = abs(pattern[0]) #Gets first frequency point
    mag_db = 20 * np.log10(magnitude)
    loss = 0 - mag_db
    print(mag_db)
    # normalize
    #pattern_norm = pattern / np.max(pattern)

    # index of target angle
    #idx_target = int(np.argmin(np.abs(angles_deg - target_angle_deg)))

    # simple loss = 1 - normalized gain at target angle
    #loss = 1.0 - pattern_norm[idx_target]
    return loss, mag_db
    
def calibration_step(v, k, vna_instance, rpi):
    """
    Perform one SPSA iteration updating (a,b,c).
    Returns updated (a,b,c) and (L_plus, L_minus).
    """
    ak = max(int(a0 / (k + 1) ** alpha), 1)  # learning rate
    ck = max(int(c0 / (k + 1) ** gamma), 1)  # perturbation magnitude
    
    # random ±1 perturbations per element
    delta_v = np.random.choice([-1.0, 1.0], size=SIZE)

    # plus / minus parameter sets
    v_plus = v + ck * delta_v
    v_minus = v - ck * delta_v
    
    v_minus = np.clip(v_minus, 0, 8191)
    v_plus = np.clip(v_plus, 0, 8191)

    # evaluate loss for each perturbed set
    L_plus, magnitude_p = compute_loss(v_plus, vna_instance, rpi)
    L_minus, magnitude_m = compute_loss(v_minus, vna_instance, rpi)

    # scalar difference
    diff = L_plus - L_minus

    # SPSA gradient estimate for each parameter:
    #   g_k = (L+ - L-) / (2 ck Δ_k)
    g_v = diff / (2.0 * ck) * (1.0 / delta_v)

    # gradient descent update: param <- param - ak * g
    v_new = v - ak * g_v

    v_new = np.clip(v_new, 0, 8191)
    
    return v_new, L_plus, L_minus, magnitude_p, ak, ck, v_plus[0][0]
    

def main():
    #v_model = np.random.randint(0,8191, SIZE)  #initially assume random voltages [0,100), half of (2^14)/2
    v_model = np.full(SIZE, 2450)
    #x = np.linspace(0,10,100)
    #plot_sinc(x, a_model[0][0], b_model[0][0], c_model[0][0])
    #From UconnDataProcessing program phases are 8x12, this returns a 12x8 transposed then flipped up/down array for Sam's code that writes to the DACs
    #target_angle_deg = read_phase_map_file(PHASE_MAP_FILE_LB)  # beam steering target includes horn cancellization and steering angle
    #print(target_angle_deg)
    lp_arr = []
    magn_arr = []
    ak_arr = []
    ck_arr = []
    v_arr = []
    vp_arr = []
    vna_instance = VnaInstance(IP_ADDR_VNA)
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
    print("Starting SPSA calibration...")
    t0 = time.time()
    for k in range(num_iters):
        print(f"Iter: {k+1}")
        v_model, Lp, Lm, magnitude_p, ak, ck, vp = calibration_step(v_model, k, vna_instance, rpi)
        lp_arr.append(Lp)
        magn_arr.append(magnitude_p)
        ak_arr.append(ak)
        ck_arr.append(ck)
        v_arr.append(v_model[0][0])
        vp_arr.append(vp)
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

    print(v_model)
    vna_instance.rm.close()
    rpi.stop_program()
    rpi.close()
    #Plot Loss
    plt.figure()
    x = np.arange(1,len(lp_arr)+1)
    plt.plot(x,lp_arr)
    plt.title("Loss_plus")
    plt.xlabel("Iteration")
    plt.ylabel("Lp")
    plt.grid()
 
    
    #Plot perterbattion scale
    plt.figure()
    plt.plot(x,ck_arr)
    plt.title("c_k")
    plt.xlabel("Iteration (k)")
    plt.ylabel("c_k")
    plt.grid()

    
    #Plot learning scale
    plt.figure()
    plt.plot(x,ak_arr)
    plt.title("a_k")
    plt.xlabel("Iteration (k)")
    plt.ylabel("a_k")
    plt.grid()

    
    #Plot magnitude received
    plt.figure()
    plt.plot(x,magn_arr)
    plt.title("Magnitude at 19.4 GHz")
    plt.xlabel("Iteration (k)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    
    #Updated voltages
    plt.figure()
    v_arr = np.array(v_arr)
    plt.plot(x, v_arr*100/8191)
    plt.title("Updated Voltage at element [0][0]")
    plt.xlabel("Iteration (k)")
    plt.ylabel("Voltage (V)")
    plt.grid()
    
    #Plot perturbed voltages
    plt.figure()
    vp_arr = np.array(vp_arr)
    plt.plot(x, vp_arr*100/8191)
    plt.title("Perturbed Voltage at element [0][0]")
    plt.xlabel("Iteration (k)")
    plt.ylabel("Voltage (V)")
    plt.grid()
    
    
    plt.show()
    
    print("Done.")
    
if __name__ == "__main__":
    main()