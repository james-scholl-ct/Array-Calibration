# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:40:55 2026

@author: labuser
"""




import numpy as np
from reflectek_tools import lcm_control
from pathlib import Path
from reflectek_tools import VNATest
import time

def main():
    
    EXP_DIR = r"C:\Users\uconn\OneDrive\Desktop\Over the air Tests\2026-09-28_SmallArray_Gimbal"
    experiment_dir = Path(EXP_DIR)
    v_arr = []
    amp_arr = []
    phase_arr = []
    
    db = lcm_control.DB()
    FreqStart = 16e9
    FreqStop = 21e9
    numPts = 1001
    #Parameters for the number of x and y coordinates
   
    
    #Parameter for the number of distinct voltages to be measered at
    # volts = 21
    # volts  = [0.0, 1.0, 1.25,1.5,1.75, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.75, 3, 3.25, 3.5, 4.0, 5.0]
    volts = np.linspace(0,9,46)
    #Create empty arrays to store all of the measurements to be taken organized by x index, y index,  and voltage bias
    comp_arr = np.empty((len(volts), numPts), dtype = np.complex128)
    amp_arr =  np.empty((len(volts), numPts))
    phase_arr =  np.empty((len(volts), numPts))
    
    # Initialize VNA Scan Parmeters
    VNATest.init(str(FreqStart), str(FreqStop), str(numPts), port="S21")
   
    #Start for loop for the number of voltages given by the parameter 'volts' defined on line 396
    for i in range(len(volts)):
        value = float(9-volts[i])
        voltages = lcm_control.make_steering_array_dualband(np.full((24, 8), value),np.full((12, 8), value))
        db.steer(voltages, 'rx')
        print(f"Voltage {value}")
        time.sleep(40)
            
        #Start for loop for the number of x coordinates given by the parameter 'xCoord' defined on line 392
        
        sdata = VNATest.trigger()#imagefile, datafile)
        phase = np.round(np.degrees(np.angle(sdata)),3)
        amp = np.round(np.abs(sdata),5)
                    
        #Store data into predifned slots from the given x, y, and voltage parameters on lines 392,393, and 396
        comp_arr[i] = sdata
        amp_arr[i] = amp
        phase_arr[i] = phase
                        
        #Save all the data collected so far for the given x and y coordinates 
        np.savez(
            experiment_dir / f'2026-09-30_{FreqStart}-{FreqStop}GHz_{numPts}.npz',  # The HBActive is with HB inactive remember to switch that back 
            comp=comp_arr,
            amplitudes=amp_arr,
            phases=phase_arr,
            volts=volts,
            iteration = i)
        time.sleep(1)

    if value >= volts[-1]:
        voltages = lcm_control.make_steering_array(np.full((12,8),9.0), 'lb')
        db.steer(voltages)
        time.sleep(1)
        print("reset to zero")
    db.shutdown()
    
if __name__ == "__main__":
    main()