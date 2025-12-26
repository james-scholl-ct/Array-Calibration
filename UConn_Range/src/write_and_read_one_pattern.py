# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 11:40:31 2025

@author: NSI-MI
"""
import numpy as np
import math
import time
import paramiko
from typing import Optional
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import gc
import win32com.client
import json
import subprocess

SCAN_FILENAME = r"C:\NSI2000\Data\Carillon\calibration_scan_real.nsi"

BEAM = 27

PI_HOST = "192.168.6.30" #IP of PI controlling DACs
USERNAME = "feix"         
PASSWORD = "password"          
KEY_FILE = None # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"

STOP_FILE = r"/home/feix/STOP.txt"
LOCAL_FILE_HB = r"C:\NSI2000\Data\Carillon\HB_voltages.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = r"C:\NSI2000\Data\Carillon\LB_voltages.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = r"/home/feix/Desktop/dataHB.csv"  # where to put it on the Pi
REMOTE_FILE_LB = r"/home/feix/Downloads/2025-12-18 VoltageMap_HornCorrection.csv"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/feix/Gen3DAC60096EVM_SPI_RPi5.py" #Location of program on PI that updates DACs
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

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

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
            
def ssh_and_update_dacs(
    host: str,
    username: str,
    password: Optional[str],
    local_file_hb: str,
    local_file_lb: str,
    remote_file_hb: str,
    remote_file_lb: str,
    remote_command: str,
    port: int = 22,
    key_filename: Optional[str] = None,
):
    """
    1. Copies local_file -> remote_file on the Pi
    2. Runs remote_command on the Pi
    3. Waits for it to finish and returns stdout, stderr, exit_status
    """

    # --- Connect over SSH ---
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    client.connect(
        hostname=host,
        port=port,
        username=username,
        password=password,
        key_filename=key_filename,
        look_for_keys=True,
    )

    # --- Copy HB file using SFTP ---
    sftp = client.open_sftp()
    print(f"Uploading {local_file_hb} -> {remote_file_hb} ...")
    sftp.put(local_file_lb, remote_file_lb)
    print("Upload complete.")
    sftp.close()
    
    # --- Copy LB file using SFTP ---
    sftp = client.open_sftp()
    print(f"Uploading {local_file_lb} -> {remote_file_lb} ...")
    sftp.put(local_file_lb, remote_file_lb)
    print("Upload complete.")
    sftp.close()

    # --- Run command on the Pi ---
    print(f"Running remote command: {remote_command}")
    
    client.exec_command(remote_command)
    #stddin, stdout, stderr = client.exec_command(remote_command)

    # IMPORTANT: wait for completion first
    #exit_status = stdout.channel.recv_exit_status()

    #out = stdout.read().decode("utf-8", errors="replace")
    #err = stderr.read().decode("utf-8", errors="replace")

    #print("STDOUT:", out)
    #print("STDERR:", err)
    #print(f"Command finished with exit code {exit_status}\n")
    
    client.close()
    
def stop_program(
    host: str,
    username: str,
    password: Optional[str],
    local_file_hb: str,
    local_file_lb: str,
    remote_file_hb: str,
    remote_file_lb: str,
    remote_command: str,
    port: int = 22,
    key_filename: Optional[str] = None,
):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
    client.connect(
        hostname=host,
        port=port,
        username=username,
        password=password,
        key_filename=key_filename,
        look_for_keys=True,
    )
        
    sftp = client.open_sftp()
    with sftp.open(STOP_FILE, "w"):
        pass
    
class NSI2000Client:
    def __init__(self, visible=True):
        self.visible = visible
        self.server = None
        self.app = None
        self.cmd = None

    def connect(self):
        # Create/attach COM server
        self.server = win32com.client.Dispatch("NSI2000.server")
        self.app = self.server.AppConnection
        self.app.Visible = self.visible
        
        # Script interface
        self.cmd = self.app.ScriptCommands
        return self

    def disconnect(self):
        # Release COM refs
        self.cmd = None
        self.app = None
        self.server = None

        gc.collect()
        
    def run_scan_get_hor_amp(self, filename):
        #Runs the scan specified in filename, gets the amplitude at all horizontal points
        #for beam 1 at the first vertical point
        #Also saves a listing of the data and test
        start_time = time.time()
        self.cmd.MEAS_CREATE_NEW_SCAN() #For some scans post-processing doesnt finish, this closes the previous scan
        self.cmd.MEAS_ACQUIRE(filename, True)
        #objNSI2000.FF_LISTING_TO_FILE(data_filename, False)
        #objNSI2000.FF_VCUT
        #print(objNSI2000.FFPOL1Array())
        nf_hpts = int(self.cmd.NF_HPTS)
        amp = np.zeros(nf_hpts)
        self.cmd.SELECT_BEAM(BEAM)
        for i in range(nf_hpts):
            #print(nsi.cmd.NFPOL1_AMP(i, j))
            amp[i] = self.cmd.NFPOL1_AMP(i, 0)[0]
                
        #while 1:
        #    print(objNSI2000.STATUS_MESSAGE())
        #    time.sleep(0.1)  # 100 ms polling

        elapsed = time.time() - start_time
        print(f"Acquisition completed in {elapsed:.1f} seconds")
        return amp
    def save_scan(self, k, is_loss_plus:bool, cal_folder):
        if (is_loss_plus):
            cal_file = cal_folder / f"cal_iter_{k}_Lp.asc"
        else:
            cal_file = cal_folder / f"cal_iter_{k}_Lm.asc"
        self.cmd.NF_LISTING_TO_FILE(cal_file)
        cal_file = cal_folder / f"cal_iter_{k}.asc"
        self.cmd.NF_LISTING_TO_FILE(cal_file)
        #Add code to also perform an hcut and save the graph
    # Context manager support: ensures cleanup 
    def __enter__(self): 
        return self.connect() 
    
    def __exit__(self, exc_type, exc, tb): 
        self.disconnect() 
        return False # don't suppress exceptions

def main():
    
    #updates low band array file on local computer
    update_lb_array_file(INIT_VOLTAGE_MAP)
    #sends low and high band array files to PI and runs remote command to update DACs
    ssh_and_update_dacs(
        host=PI_HOST,
        username=USERNAME,
        password=PASSWORD,
        local_file_hb=LOCAL_FILE_HB,
        local_file_lb=LOCAL_FILE_LB,
        remote_file_hb=REMOTE_FILE_HB,
        remote_file_lb=REMOTE_FILE_LB,
        remote_command=REMOTE_COMMAND,
        key_filename=KEY_FILE,
    )
    with NSI2000Client() as nsi:
        pattern = nsi.run_scan_get_hor_amp(SCAN_FILENAME)
    
    stop_program(
        host=PI_HOST,
        username=USERNAME,
        password=PASSWORD,
        local_file_hb=LOCAL_FILE_HB,
        local_file_lb=LOCAL_FILE_LB,
        remote_file_hb=REMOTE_FILE_HB,
        remote_file_lb=REMOTE_FILE_LB,
        remote_command=REMOTE_COMMAND,
        key_filename=KEY_FILE,
    )
    #Plot magnitude vs span
    fig1, ax1 = plt.subplots()
    ax1.plot(pattern)
    ax1.set_title("Magnitude vs span")
    ax1.set_xlabel("span: -2.5 to 2.5 in")
    ax1.set_ylabel("Magnitdue (dB)")
    ax1.grid()
    plt.close(fig1)
if __name__ == "__main__":
    main()