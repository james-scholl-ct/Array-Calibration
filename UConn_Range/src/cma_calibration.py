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

LC_DELAY_TIME = 40 #in secs

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

#Set the beam (corresponds to frequency measured) number that you put in NSI software. 19.3 Ghz is ideal for low band
BEAM = 1

horn_inverse = np.array([
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

#unmap paramaters from ideally 0 to 10
def coord_map(x):
    x_a, x_k, x_v0, x_y0 = x
    y0max = -150
    y0min = 0
    amax = 360
    amin = 0
    kmax = 100 #k must be negative, negate later
    kmin = 0.01
    vmax = 10.5
    vmin = 0
    y0 = y0min + (y0max-y0min)*(x_y0/10)
    a = amin + (amax-amin)*(x_a/10)
    k = -(kmin * (kmax/kmin)**(x_k/10))
    v0 = vmin + (vmax-vmin)*(.5+.5*np.tanh(.8*(x_v0-5)))
    return np.array([a, k, v0, y0])

def coord_map_inverse(p):
    a, k, v0, y0 = p

    y0max = -150
    y0min = 0

    amax = 360
    amin = 0

    kmax = 100
    kmin = 0.01

    vmax = 10.5
    vmin = 0

    # invert a
    x_a = 10 * (a - amin) / (amax - amin)

    # invert y0
    x_y0 = 10 * (y0 - y0min) / (y0max - y0min)

    # invert k (log-mapped, negative)
    x_k = 10 * np.log(np.abs(k) / kmin) / np.log(kmax / kmin)

    # invert v0 (tanh)
    z = 2 * (v0 - vmin) / (vmax - vmin) - 1
    z = np.clip(z, -0.999999, 0.999999)   # numerical safety
    x_v0 = 5 + np.arctanh(z) / 0.8

    return np.array([x_a, x_k, x_v0, x_y0])


def phase_from_voltage(v, y0, A, k, v0):
    """
    V  : voltage in [0,10]
    y0 : offset
    A  : amplitude
    k  : slope
    v0 : midpoint
    """
    phi = y0 + A / (1.0 + np.exp(-k*(v - v0)))
    return phi

def voltage_from_phase(phi, params, eps=1e-4):
    A, k, v0, y0 = params

    q = (phi - y0) / A
    q = np.clip(q, eps, 1 - eps)

    V = v0 + (1.0 / k) * np.log(q / (1 - q))
    
    return np.clip(V, 0.0, 10.5)


def objective(x):
    try:
        params = coord_map(x) 
        voltages = voltage_from_phase(horn_inverse, params)
        update_lb_array_file(voltages)
        #sends low and high band array files to PI and runs remote command to update DACs
        rpi.update_dacs()
        time.sleep(LC_DELAY_TIME)
        pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)
        print(pattern)
        return pattern  # maximize magnitude
    except Exception:
        return -1e9


sigma0 = 2          # explore ~20% of full scale at first
starting_sigmoid = np.array([227.32021921, -0.6683808, 1.77826861, -53.15161643])
starting_params = coord_map_inverse(starting_sigmoid)

opts = {
    "popsize": 20,
    "maxfevals": 400,
    "verb_disp": 1,
}

es = cma.CMAEvolutionStrategy(starting_params, sigma0, opts)

while not es.stop():
    X = es.ask()
    F = [objective(x) for x in X]
    es.tell(X, F)
    es.disp()

best_x = np.array(es.result.xbest)        # internal variables
best_p = coord_map(best_x)                # physical parameters
best_f = es.result.fbest