# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:17 2025
Sends 8x12 HB voltages at 27.2Ghz to 0-100V array
@author: SchollJamesAC3CARILL

"""
import pyvisa
import numpy as np
import math
import time
import paramiko
from typing import Optional
import matplotlib.pyplot as plt


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

class PiController:
    """
    Connects to a raspberry pi via ssh with Paramiko. Copies a local low and high band file to the files on the PI. 
    Runs the remote command that starts the python program on the PI which updates the DACs via SPI.
    Creates a stop text file that is watched for by that program to stop it so that it can read another set of HB and LB voltages
    #Reopens VNA connection each time, closes resource manager at the very end
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

class VnaInstance:
    """
    In NSI Max software on VNA, in system settings, make sure System Configuration Web Access is set to Local and Remote
    """
    def __init__(self, ip_addr):
        self.ip_addr = ip_addr
        self.rm = pyvisa.ResourceManager()
        self.instr = None
    def connect(self):
        self.instr = self.rm.open_resource(self.ip_addr)
        self.instr.timeout = 60000 #60 seconds
        print("VNA ID:", self.instr.query("*IDN?").strip())
    def disconnect(self):
        if self.instr is not None:
            try:
                self.instr.close()
            except:
                pass
            self.instr = None
    def sweep(self, start, stop, points):
        """
        Returns an array of size points containing complex data, the first value corresponds to first measured frequency
        """
        self.connect()
        #instr.write('INIT:IMM; *WAI')
        #self.instr.write(':SENS:HOLD:FUNC HOLD')
        #self.instr.write(':TRIG:SING')
        try: 
            self.instr.write('LSB;FMB') 
            self.instr.write(f'SENS1:FREQ:START {start}')
            self.instr.write(f'SENS1:FREQ:STOP {stop}')
            self.instr.write(f'SENS1:SWE:POIN {points}')
            self.instr.write(':CALC1:PAR1:DEF S21')
            #self.instr.write('INIT:IMM')
            self.instr.write(':SENS:HOLD:FUNC HOLD')
            self.instr.write(':TRIG:SING')
            self.instr.query('*OPC?') #operation complete? -Waits for operation complete
            print("Querying data...")
            sdata = self.instr.query_binary_values(':CALC1:DATA:SDAT?', datatype = 'd', container = np.array).reshape((-1,2))  # any data query
            sdata = sdata[:,0] + sdata[:,1]*1j
            print("Received response.\n")
            return sdata
        except pyvisa.errors.VisaIOError as e:
            print("VNA communication error:", e)
            self.disconnect()
            raise
        finally:
            self.disconnect()


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