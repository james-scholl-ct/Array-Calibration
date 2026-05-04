# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:26:36 2026

@author: NSI-MI
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import subprocess
import argparse
import csv
import math
from Shared.PiController import PiController
from Shared.NSI2000Client import NSI2000Client
from Shared.VnaInstance import VnaInstance


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
            
def update_hb_array_file(V):
    #V = np.round(V * DAC_MIN_STEP_SIZE, 3)
    #This program is for HB only, so create a 0V array for the low band
    with open(LOCAL_FILE_LB, "w") as f:
        for row in np.zeros((12,8)):
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")
    with open(LOCAL_FILE_HB, "w") as f:
        for row in V:
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")

def measure(vna, start, stop, points):
    
    data = vna.sweep(start, stop, points)
    data_db = 20 * np.log10(np.abs(data))
    print(max(data_db))
    return  data_db

def shutdown(rpi, vna):
        #voltages = np.zeros((24,8))
        #update_hb_array_file(voltages)
        #rpi.update_dacs()
        time.sleep(1)
        #rpi.stop_program()
        rpi.close()
        vna.disconnect()
voltages = np.array([
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
def get_data():
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
        vna=VnaInstance("TCPIP0::192.168.6.150::inst0::INSTR")
        vna.connect()
        folder = Path(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\MagVsFreq")
        savefile = folder/"magvsfreq_theta0_phi0_crosspol.npz"
        step_size = 50e6
        start = 17e9
        stop = 38e9
        frequency = np.arange(start, stop+1, step_size)
        points = len(frequency)
        frequency_ghz = frequency/1e9
        #update_hb_array_file(voltages)
        #time.sleep(40)
        #rpi.update_dacs()
        try:
            data = measure(vna, start, stop, points)
        except Exception:
            shutdown(rpi, vna)
        plt.figure()
        plt.plot(frequency_ghz, data)
        #plt.plot(elapsed_time[-1], data[-1], 'ro')
        plt.xlabel("Frequency (GHz)")
        plt.ylim(-110, -20)
        #plt.xlim(start/1e9, stop/1e9)
        plt.xticks(np.arange(start/1e9, stop/1e9+1, 1))
        plt.ylabel("Magnitude (dB)")
        plt.title("Magnitude vs Frequency Cross-Pol for Beam at az=0°, el=0°")
        plt.show()
        np.savez(savefile, mag=data, freq=frequency)
        shutdown(rpi, vna)
    
def plot_data():
    file00 = np.load(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\MagVsFreq\magvsfreq_theta0_phi0_copol.npz")
    filecross = np.load(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\MagVsFreq\magvsfreq_theta0_phi0_crosspol.npz")
    data00 = file00["mag"]
    freq00 = file00["freq"]
    datacross = filecross["mag"]
    freqcross = filecross["freq"]

    freq_ghz = freq00/1e9
    start = freq_ghz[0]
    stop = freq_ghz[-1]

    plt.figure()
    plt.plot(freq_ghz, data00, label="Co-Pol")
    plt.plot(freq_ghz, datacross, label="Cross-Pol")
    plt.xlabel("Frequency (GHz)")
    plt.ylim(-110, -20)
    plt.xticks(np.arange(start, stop+1, 1))
    plt.grid()
    plt.legend()
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude vs Frequency for Beam at az=0°, el=0°")
    plt.show()
    
#get_data()
plot_data()