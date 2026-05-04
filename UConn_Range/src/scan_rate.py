# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:09:01 2026

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

Z_DIS = 43.75 #scanner distance from AUT inches


def compute_y(theta_deg, phi_deg, R):

    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)

    numerator = math.sin(theta) * math.sin(phi)
    denominator = math.sqrt(
        (math.sin(theta)**2) * (math.cos(phi)**2) +
        (math.cos(theta)**2)
    )

    return R * numerator / denominator

def angle_str(value: float) -> str:
    """Format angles so 20.0 -> '20' and 20.5 -> '20.5'."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)

def make_filename(theta: float, phi: float) -> str:
    return f"voltages_theta{angle_str(theta)}_phi{angle_str(phi)}.csv"


def load_csv_numbers(path: Path) -> list[list[float]]:
    """Load a CSV containing rows of numbers."""
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    return rows

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

def measure(vna):
    data = vna.sweep(28.2e9, 28.3e9, 1)[0]
    return  20 * np.log10(np.abs(data))

def wait_until_stable(nsi, rpi, vna, start, poll_interval=.1, threshold_fraction=.01, timeout = 60, base_to_target=True):
    def do_check(check):
        if base_to_target == True:
            check_val = -.1
            if check < check_val:
                print(f"true check:{check}")
                return True
            else:
                return False
        else:
            check_val = .1
            if check > check_val:
                print(f"true check:{check}")
                return True
            else:
                return False
    measurments = []
    elapsed_time = []
    start_checking = False
    prev_value = None
    initial=None
    value = None
    i = 0
    start_timeout = time.perf_counter()
    while True:
        check_timeout(nsi, rpi, vna, start_timeout, "Measurement did not stabilize in time", timeout)
        i = i + 1
        now = time.perf_counter()
        value = measure(vna)
        print(f"Value:{value}")
        #after_meas = time.perf_counter()
        #meas_time = after_meas-now
        #print(meas_time)
        elapsed = now-start
        measurments.append(value)
        elapsed_time.append(elapsed)
        if prev_value is not None and start_checking == False:
            check =  (abs(value) - abs(initial))/ abs(initial)
            start_checking = do_check(check)
        if prev_value is not None and start_checking==True:
            frac_change = abs(value - prev_value) / abs(prev_value)
            if frac_change < threshold_fraction:
                return now-start_timeout, elapsed_time, measurments
        prev_value = value
        if i ==1:
            initial = value
        time.sleep(poll_interval)
            
def shutdown(nsi, rpi, vna):
        voltages = np.zeros((24,8))
        update_hb_array_file(voltages)
        rpi.update_dacs()
        time.sleep(1)
        nsi.cmd.MOVE_TO_SCANNER_ZERO()
        nsi.disconnect()
        rpi.stop_program()
        rpi.close()
        vna.disconnect()
        
def move_position(nsi, rpi, vna, x,y):
    start = time.perf_counter()
    moved_x = False
    moved_y = False
    accuracy = .001 #.1%
    nsi.move_x(x)
    time.sleep(1)
    while moved_x == False:
        check_timeout(nsi, rpi, vna, start, "x position never stabilized", 60)
        if abs(nsi.get_haxis_pos() - x)  < accuracy:
            moved_x = True
        time.sleep(1)
    nsi.move_y(y)
    time.sleep(1)
    while moved_y == False:
        check_timeout(nsi, rpi, vna, start, "y position never stabilized", 60)
        if abs(nsi.get_vaxis_pos() -y) < accuracy:
            moved_y = True
        time.sleep(1)
    
def check_timeout(nsi, rpi, vna, start, message, timeout):
    if time.perf_counter() - start > timeout:
        shutdown(nsi, rpi, vna)
        raise TimeoutError(message)
        
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--base_theta", type=float, required=True)
    scan_parser.add_argument("--base_phi", type=float, required=True)
    scan_parser.add_argument("--move_theta", type=float, required=True)
    scan_parser.add_argument("--move_phi", type=float, required=True)
    scan_parser.add_argument(
        "--folder",
        type=Path,
        default=Path(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\sanford_optimization\Beams\Ga\High Band\Feed bottom of array\28.2Ghz"),
        help="Folder containing the CSV files",
    )
    scan_parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of rise and fall times",
    )

    args = parser.parse_args()

    if args.command == "scan":
        base_file = args.folder / make_filename(args.base_theta, args.base_phi)
        move_file = args.folder / make_filename(args.move_theta, args.move_phi)

        print(f"Looking for base file: {base_file}")
        print(f"Looking for move file: {move_file}")

        if not base_file.exists():
            print(f"ERROR: Base file not found: {base_file}")
            return 1

        if not move_file.exists():
            print(f"ERROR: Move file not found: {move_file}")
            return 1

        y_move = compute_y(args.move_theta, args.move_phi, Z_DIS)
        print(f"y_move = {y_move}")
        base_numbers = load_csv_numbers(base_file)
        move_numbers = load_csv_numbers(move_file)

        print(f"Loaded base file with {len(base_numbers)} rows")
        print(f"Loaded move file with {len(move_numbers)} rows")

        print("\nBase data:")
        print(base_numbers)

        print("\nMove data:")
        print(move_numbers)
        
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
        vna=VnaInstance("TCPIP0::192.168.6.150::inst0::INSTR")
        vna.connect()
        
        frequency = args.folder.name
        
        threshold_fraction = .001
        
        scan_directory = args.folder / "switching_speed"
        scan_directory.mkdir(parents=True, exist_ok=True)
        scan_save_file = scan_directory / f"{args.base_theta}_{args.base_phi}_to_{args.move_theta}_{args.move_phi}_{frequency}_stablefrac_{threshold_fraction}_iters_{args.iters}.npz"
        
        
        move_position(nsi, rpi, vna, args.move_theta, y_move)
        
        update_hb_array_file(base_numbers)
        rpi.update_dacs()
        time.sleep(10)
        print("Finished base beam")
        square = []
        time_arr = []
        start = time.perf_counter()
        up_stable = []
        down_stable = []
        for i in range(args.iters):
            update_hb_array_file(move_numbers)
            rpi.update_dacs()
            #time.sleep(40)
            elapsed, elapsed_time, data = wait_until_stable(nsi, rpi, vna, start, poll_interval=.1, threshold_fraction = threshold_fraction, base_to_target = True)
            up_stable.append(elapsed)
            square.append(data)
            time_arr.append(elapsed_time)
            update_hb_array_file(base_numbers)
            rpi.update_dacs()
            #time.sleep(40)
            elapsed, elapsed_time, data = wait_until_stable(nsi, rpi,vna,start, poll_interval=.1, threshold_fraction = threshold_fraction, base_to_target = False)
            down_stable.append(elapsed)
            square.append(data)
            time_arr.append(elapsed_time)
        mean_up = round(np.average(up_stable),3)
        mean_down = round(np.average(down_stable),3)
        square = np.concatenate(square)
        time_arr = np.concatenate(time_arr)
        plt.figure()
        plt.plot(time_arr, square, '-o')
        #plt.plot(elapsed_time[-1], data[-1], 'ro')
        plt.xlabel("Time (s)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"Magnitude at ({args.move_theta}, {args.move_phi}) deg switching from/to ({args.base_theta}, {args.base_phi}) deg. {frequency}. Rise stable avg: {mean_up}s, fall stable avg: {mean_down}s")
        plt.show()
        #nsi.save_scan(r"C:\NSI2000\Data\Carillon\temp.asc")
        np.savez(scan_save_file, mag= square, times = time_arr)
        shutdown(nsi, rpi, vna)
    
if __name__ == "__main__":
    main()