# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:17 2025

@author: SchollJamesAC3CARILL
192 element high band
24x8
"""
import pyvisa
import numpy as np
import math
import time
import paramiko
from typing import Optional
import matplotlib.pyplot as plt


PHASE_MAP_FILE_LB = rf"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\phases_with_beam_steering_0theta_0phi_hex_12x8.txt" 

PI_HOST = "raspberrypi.local" #IP of PI controlling DACs
USERNAME = "carillon"         
PASSWORD = "carillon"          
KEY_FILE = None               # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"

LOCAL_FILE_HB = rf"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\sshTest.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = rf"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\sshTest2.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = "/home/carillon/calibration/sshTest.txt"  # where to put it on the Pi
REMOTE_FILE_LB = "/home/carillon/calibration/sshTest2.txt"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/carillon/calibration/test.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"python3 {REMOTE_PROGRAM}"

IP_ADDR_VNA = "TCPIP0::192.168.6.150::inst0::INSTR" #VNA IP addr
#VNA sweep params
START = 16e9
STOP = 16.1e9
POINTS = 2

SIZE = (12,8)

DAC_MIN_STEP_SIZE = 20/4096

# SPSA hyperparameters
a0 = 30   # learning-rate scale in dac steps
c0 = 60  # perturbation scale in DAC steps should be 2-5x a0
alpha = 0.6 #.6-.8
gamma = 0.1
num_iters = 10

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

    print("STDOUT:", out)
    print("STDERR:", err)
    print(f"Command finished with exit code {exit_status}\n")
    
    client.close()

class VnaInstance:
    def __init__(self, ip_addr):
        self.ip_addr = ip_addr
        self.rm = None
        self.instr = None
    def connect(self):
        self.rm = pyvisa.ResourceManager()
        self.instr = self.rm.open_resource(self.ip_addr)
        self.instr.timeout = 10000
    def disconnect(self):
        self.instr.close()
        self.rm.close()
    def sweep(self, start, stop, points):
        print(self.instr.query("*IDN?").strip())
        self.instr.write('LSB;FMB') 
        self.instr.write('SENS1:FREQ:START ',str(START))
        self.instr.write('SENS1:FREQ:STOP ',str(STOP))
        self.instr.write(':SENS1:SWE:POIN ',str(POINTS))
        self.instr.write(':CALC1:PAR1:DEF S21')
        
        #instr.write('INIT:IMM; *WAI')
        self.instr.write(':SENS:HOLD:FUNC HOLD')
        self.instr.write(':TRIG:SING')
        # Query
        print("Querying...")
        sdata = self.instr.query_binary_values(':CALC1:DATA:SDAT?', datatype = 'd', container = np.array).reshape((-1,2))  # any data query
        sdata = sdata[:,0] + sdata[:,1]*1j
        print("Received response.\n")
        return sdata


def get_pattern(vna_instance):
    pattern = vna_instance.sweep(START, STOP, POINTS)
    return pattern


def compute_loss(v, vna_instance):
    """
    This is what you'd do with real hardware:
      1. send V to array
      2. measure pattern
      3. compute scalar loss

    Here we simulate the pattern.
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
    
    pattern = get_pattern(vna_instance)
    magnitude = abs(pattern[0])
    loss = 10-magnitude

    # normalize
    #pattern_norm = pattern / np.max(pattern)

    # index of target angle
    #idx_target = int(np.argmin(np.abs(angles_deg - target_angle_deg)))

    # simple loss = 1 - normalized gain at target angle
    #loss = 1.0 - pattern_norm[idx_target]
    return loss, magnitude
    
def calibration_step(v, k, vna_instance):
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

    # evaluate loss for each perturbed set
    L_plus, magnitude_p = compute_loss(v_plus, vna_instance)
    L_minus, magnitude_m = compute_loss(v_minus, vna_instance)

    # scalar difference
    diff = L_plus - L_minus

    # SPSA gradient estimate for each parameter:
    #   g_k = (L+ - L-) / (2 ck Δ_k)
    g_v = diff / (2.0 * ck) * (1.0 / delta_v)

    # gradient descent update: param <- param - ak * g
    v_new = v - ak * g_v

    v_new = np.clip(v_new, 0, 2095)
    
    return v_new, L_plus, L_minus, magnitude_p, ak, ck, v_new[0][0]
    

def main():
    v_model = np.random.randint(0,2095, SIZE)  #initially assume random voltages [0,10)
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
    vna_instance = VnaInstance(IP_ADDR_VNA)
    vna_instance.connect()
    print("Starting SPSA calibration...")
    t0 = time.time()
    for k in range(num_iters):
        print(f"Iter: {k+1}")
        v_model, Lp, Lm, magnitude_p, ak, ck, v_new = calibration_step(v_model, k, vna_instance)
        lp_arr.append(Lp)
        magn_arr.append(magnitude_p)
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

    vna_instance.disconnect()
    
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
    
    #Plot perturbed voltages
    plt.figure()
    v_arr = np.array(v_arr)
    plt.plot(x, v_arr*10/2095)
    plt.title("Perturbed Voltage at element [0][0]")
    plt.xlabel("Iteration (k)")
    plt.ylabel("Voltage (V)")
    plt.grid()
    
    plt.show()
    
    print("Done.")
    
if __name__ == "__main__":
    main()