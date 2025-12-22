# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:17 2025

@author: SchollJamesAC3CARILL
192 element high band
24x8
"""

import numpy as np
import math
import time
import paramiko
from typing import Optional
#import matplotlib
#matplotlib.use("Qt5Agg")   # or "TkAgg"
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import gc
import win32com.client

CAL_DIRECTORY = r"C:\NSI2000\Data\Carillon"
SCAN_FILENAME = r"C:\NSI2000\Data\Carillon\calibration_scan_real.nsi"

#PHASE_MAP_FILE_LB = r"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\phases_with_beam_steering_0theta_0phi_hex_12x8.txt" 

PI_HOST = "192.168.6.30" #IP of PI controlling DACs
USERNAME = "feix"         
PASSWORD = "password"          
KEY_FILE = None               # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"

LOCAL_FILE_HB = r"C:\NSI2000\Data\Carillon\HB_voltages.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = r"C:\NSI2000\Data\Carillon\LB_voltages.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = r"/home/feix/Desktop/dataHB.csv"  # where to put it on the Pi
REMOTE_FILE_LB = r"/home/feix/Downloads/2025-12-18 VoltageMap_HornCorrection.csv"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/feix/Gen3DAC60096EVM_SPI_RPi5.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"python3 {REMOTE_PROGRAM}"

#IP_ADDR_VNA = "TCPIP0::192.168.6.150::inst0::INSTR" #VNA in new haven lab IP addr
#VNA sweep params
START = 16e9
STOP = 16.1e9
POINTS = 2

INIT_VOLTAGE_MAP = np.array([ #From Sams 0-100V data and expected phase at 0,0 az and el interpolated already transposed and flipped
    [1.1554, 1.1554, 1.1554, 1.3384, 1.5020, 1.5020, 1.3869, 1.2366],
    [5.8092, 5.8092, 5.8092, 1.1554, 1.1554, 1.1554, 1.1554, 1.1554],
    [4.5520, 4.9869, 5.4874, 5.8092, 5.8092, 5.8092, 5.8092, 5.6609],
    [3.3907, 3.6844, 3.8934, 4.0859, 4.2236, 4.2414, 4.2414, 4.1366],
    [2.6797, 2.9363, 3.1225, 3.2157, 3.2826, 3.2961, 3.2290, 3.1624],
    [1.3869, 1.8255, 2.0536, 2.2612, 2.3668, 2.3668, 2.3816, 2.2917],
    [1.1554, 1.1554, 1.1554, 1.1554, 1.1554, 1.1554, 1.1554, 1.1554],
    [5.0152, 5.7577, 5.8092, 5.8092, 5.8092, 5.8092, 5.8092, 5.8092],
    [3.7723, 4.0364, 4.2776, 4.4097, 4.5309, 4.5520, 4.4492, 4.3520],
    [2.6245, 2.8963, 3.0560, 3.2157, 3.2961, 3.3095, 3.3230, 3.2424],
    [1.4570, 1.8255, 2.0536, 2.1515, 2.2458, 2.2612, 2.1833, 2.1030],
    [5.8092, 5.8092, 1.1554, 1.1554, 1.1554, 1.1554, 1.1554, 1.1554],
], dtype=float)

SIZE = (12,8)
LC_DELAY_TIME = 40 #in secs

DAC_MIN_STEP_SIZE = 20/4096


# SPSA hyperparameters
a0 = 300   # learning-rate scale in dac steps
c0 = 600  # perturbation scale in DAC steps should be 2-5x a0
alpha = 0.4 #.6-.8
gamma = 0.1
num_iters = 1000

class LiveLossPlot:
    def __init__(self, title="SPSA Loss", ylabel="loss"):
        plt.ion()  # interactive on
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel(ylabel)
        self.line, = self.ax.plot([], [])  # empty line
        self.losses = []
        self.iters = []

        self.ax.grid(True)
        self.fig.show()

    def update(self, k, loss):
        self.iters.append(k)
        self.losses.append(float(loss))

        self.line.set_data(self.iters, self.losses)

        # Rescale axes to fit new data
        self.ax.relim()
        self.ax.autoscale_view()

        # Make GUI breathe
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # tiny pause = allows GUI event loop

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
    
    stddin, stdout, stderr = client.exec_command(remote_command)

    # IMPORTANT: wait for completion first
    exit_status = stdout.channel.recv_exit_status()

    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")

    #print("STDOUT:", out)
    print("STDERR:", err)
    print(f"Command finished with exit code {exit_status}\n")
    
    client.close()

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
        
    def run_scan_get_hor_amp(self, filename, k, is_loss_plus:bool, cal_folder):
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
        self.cmd.SELECT_BEAM(1)
        for i in range(nf_hpts):
            #print(nsi.cmd.NFPOL1_AMP(i, j))
            amp[i] = self.cmd.NFPOL1_AMP(i, 0)[0]
                
        #while 1:
        #    print(objNSI2000.STATUS_MESSAGE())
        #    time.sleep(0.1)  # 100 ms polling

        elapsed = time.time() - start_time
        if (is_loss_plus):
            cal_file = cal_folder / f"cal_iter_{k}_Lp.asc"
        else:
            cal_file = cal_folder / f"cal_iter_{k}_Lm.asc"
        self.cmd.NF_LISTING_TO_FILE(cal_file)
        print(f"Acquisition completed in {elapsed:.1f} seconds")
        
        cal_folder = Path(CAL_DIRECTORY)
        cal_file = cal_folder / f"cal_iter_{k}.asc"
        self.cmd.NF_LISTING_TO_FILE(cal_file)
        
        return amp
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

def compute_loss(v, vna_instance, k, is_loss_plus, cal_folder):
    """
    """
    #updates low band array file on local computer
    update_lb_array_file(v)
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
    time.sleep(LC_DELAY_TIME)
    pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, k, is_loss_plus, cal_folder)
    
    loss = loss_center_vs_sidelobes_db(pattern, 15, 2, 4)

    return loss, pattern
    
def calibration_step(v, k, vna_instance, cal_folder):
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
    
    v_minus = np.clip(v_minus, 0, 2047)

    # evaluate loss for each perturbed set
    L_plus, pattern_plus = compute_loss(v_plus, vna_instance, k, True, cal_folder)
    L_minus, pattern_minus = compute_loss(v_minus, vna_instance, k, False, cal_folder)

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
    #x = np.linspace(0,10,100)
    #plot_sinc(x, a_model[0][0], b_model[0][0], c_model[0][0])
    #From UconnDataProcessing program phases are 8x12, this returns a 12x8 transposed then flipped up/down array for Sam's code that writes to the DACs
    #target_angle_deg = read_phase_map_file(PHASE_MAP_FILE_LB)  # beam steering target includes horn cancellization and steering angle
    #print(target_angle_deg)
    lp_arr = []
    pattern_point_arr = []
    all_patterns = []
    ak_arr = []
    ck_arr = []
    v_arr = []
    #plotter = LiveLossPlot()
    nsi = NSI2000Client().connect()
    cal_path = Path(CAL_DIRECTORY)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cal_folder = cal_path / f"Calibration_{ts}"
    cal_folder.mkdir(parents = True, exist_ok = False)
    print("Starting SPSA calibration...")
    t0 = time.time()
    for k in range(num_iters):
        print(f"Iter: {k+1}")
        v_model, Lp, Lm, pattern, ak, ck, v_new = calibration_step(v_model, k, nsi, cal_folder)
        lp_arr.append(Lp)
        pattern_point_arr.append(pattern[18])
        all_patterns.append(pattern)
        ak_arr.append(ak)
        ck_arr.append(ck)
        v_arr.append(v_new)
        t1 = time.time()
        #try:
            #plotter.update(k, Lp)
        #except Exception:
        #    pass
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

    
    #Plot magnitude received in center
    plt.figure()
    plt.plot(x,pattern_point_arr)
    plt.title("Magnitude at 19.4 GHz")
    plt.xlabel("Iteration (k)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    
    #Plot magnitude vs span for last iteration
    plt.figure()
    plt.plot(all_patterns[num_iters-1])
    plt.title("Magnitude vs span")
    plt.xlabel("span: -2.5 to 2.5 in")
    plt.ylabel("Magnitdue (dB)")
    plt.grid()
    
    #Plot perturbed voltages
    plt.figure()
    v_arr = np.array(v_arr)
    plt.plot(x, np.round(v_arr*10/2047,3))
    plt.title("Perturbed Voltage at element [0][0]")
    plt.xlabel("Iteration (k)")
    plt.ylabel("Voltage (V)")
    plt.grid()

    
    plt.show()
    
    print("Done.")
    
if __name__ == "__main__":
    main()
