# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:17 2025

@author: SchollJamesAC3CARILL
96 Element LB reflectarray calibration

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

LC_DELAY_TIME = 1 #in secs

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
a0 = 300   # learning-rate scale in dac steps
c0 = 600  # perturbation scale in DAC steps should be 2-5x a0
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
            
class PiController:
    """
    Connects to a raspberry pi via ssh with Paramiko. Copies a local low and high band file to the files on the PI. 
    Runs the remote command that starts the python program on the PI which updates the DACs via SPI.
    Creates a stop text file that is watched for by that program to stop it so that it can read another set of HB and LB voltages
    """
    def __init__(
        self,
        host: str,
        username: str,
        password: Optional[str],
        local_file_hb: str,
        local_file_lb: str,
        remote_file_hb: str,
        remote_file_lb: str,
        remote_command: str,
        port: int,
        key_filename: Optional[str],
        stop_file: str,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.local_file_hb = local_file_hb
        self.local_file_lb = local_file_lb
        self.remote_file_hb = remote_file_hb
        self.remote_file_lb = remote_file_lb
        self.remote_command = remote_command
        self.port = port
        self.key_filename = key_filename
        self.stop_file = stop_file
        self.client = None
        
    def connect(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            key_filename=self.key_filename,
            look_for_keys=True,
        )
        self.client = client
        return self
    
    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
                
    def remove_stop_file(self):
        sftp = self.client.open_sftp()
        try:
            sftp.remove(self.stop_file)
            print("Stop File removed")
        except FileNotFoundError:
            pass
        sftp.close()
        
    def stop_program(self):
        sftp = self.client.open_sftp()
        with sftp.open(self.stop_file, "w"):
            pass
        sftp.close()
        
    def upload_lb_and_hb_files(self):
        sftp = self.client.open_sftp()
        print(f"Uploading {self.local_file_hb} -> {self.remote_file_hb} ...")
        sftp.put(self.local_file_hb, self.remote_file_hb)
        print("Upload complete.")
        
        print(f"Uploading {self.local_file_lb} -> {self.remote_file_lb} ...")
        sftp.put(self.local_file_lb, self.remote_file_lb)
        print("Upload complete.")
        sftp.close()
        
    def run_remote_command(self, wait: bool = False, get_pty: bool = True):
        print(f"Running remote command: {self.remote_command}")
        stdin, stdout, stderr = self.client.exec_command(self.remote_command, get_pty=get_pty)
        if not wait:
            return None
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return out, err, exit_status
    def update_dacs(self):
        self.stop_program()
        time.sleep(1)
        self.remove_stop_file()
        self.upload_lb_and_hb_files()
        self.run_remote_command()
        
    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()
    

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

def loss_center_vs_sidelobes_db(
    amp_db,
    center_idx: int,
    main_half_width: int = 2,
    guard_half_width: int = 10,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-15,
):
    """
    SPSA-friendly scalar loss using VNA amplitude in dB (dB magnitude).

    Goal: maximize energy near known center_idx, minimize energy elsewhere (sidelobes).

    Loss = -alpha * log(E_main) + beta * log(E_side)

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

    # stabilize logs
    E_main = max(E_main, eps)
    E_side = max(E_side, eps)

    loss = (-alpha * np.log(E_main)) + (beta * np.log(E_side))
    
    return loss

def compute_loss(v, vna_instance, rpi, k, is_loss_plus, cal_folder):
    """
    """
    #updates low band array file on local computer
    update_lb_array_file(v)
    
    #sends low and high band array files to PI and runs remote command to update DACs
    rpi.update_dacs()
    
    time.sleep(LC_DELAY_TIME)
    
    pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME)
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
    
    return v_new, L_plus, L_minus, pattern_plus, ak, ck, v_new[0][0]
    

def main():
    #v_model = np.random.randint(0,2095, SIZE)  #initially assume random voltages [0,10)
    v_model = np.clip(np.round(INIT_VOLTAGE_MAP/DAC_MIN_STEP_SIZE), 0, 2047)
    lp_arr = []
    pattern_point_arr = []
    all_patterns = []
    ak_arr = []
    ck_arr = []
    v_arr = []
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
    for k in range(num_iters):
        print(f"Iter: {k+1}")
        v_model, Lp, Lm, pattern, ak, ck, v_new = calibration_step(v_model, k, nsi, rpi, raw_folder)
        lp_arr.append(Lp)
        pattern_point_arr.append(pattern[18])
        all_patterns.append(pattern)
        ak_arr.append(ak)
        ck_arr.append(ck)
        v_arr.append(v_new)
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

    nsi.disconnect()
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
            "guard_band_half_width": GUARD_BAND_HALF_WIDTH
        }
    
    with open(exp_folder / "params.json", "w") as f:
        json.dump(params, f, indent=2)
        
    np.savez(
        exp_folder / "results.npz",
        final_voltages= v_new,
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
    fig3.savefig(plot_dir / "c_kVsIter.png", dpi=200)
    
    #Plot magnitude received in center
    fig4, ax4 = plt.subplots()
    ax4.plot(x,pattern_point_arr)
    ax4.set_title("Magnitude at 19.4 GHz")
    ax4.set_xlabel("Iteration (k)")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.grid()
    fig4.savefig(plot_dir / "MagInCenterVsIter.png", dpi=200)
    
    #Plot magnitude vs span for last iteration
    fig5, ax5 = plt.subplots()
    ax5.plot(all_patterns[num_iters-1])
    ax5.set_title("Magnitude vs span")
    ax5.set_xlabel("span: -2.5 to 2.5 in")
    ax5.set_ylabel("Magnitdue (dB)")
    ax5.grid()
    fig5.savefig(plot_dir / "FinalMagVsSpan.png", dpi=200)
    
    #Plot perturbed voltages
    fig6, ax6 = plt.subplots()
    v_arr = np.array(v_arr)
    ax6.plot(x, np.round(v_arr*10.5/2047,3))
    ax6.set_title("Perturbed Voltage at element [0][0]")
    ax6.set_xlabel("Iteration (k)")
    ax6.set_ylabel("Voltage (V)")
    ax6.grid()
    fig6.savefig(plot_dir / "PerturbedVoltageAt00VsIter.png", dpi=200)
    
    plt.show()
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    
    print("Done.")
    
if __name__ == "__main__":
    main()