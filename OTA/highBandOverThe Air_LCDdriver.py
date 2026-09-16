# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:40:55 2026

@author: labuser
"""




import numpy as np
from Shared import lcm_control
from pathlib import Path
import VNATest
import time


def main():
    
    EXP_DIR = r"C:\Users\labuser\Gen3WGSDualPatchGraphs\AutomationTesting"
    experiment_dir = Path(EXP_DIR)
    v_arr = []
    amp_arr = []
    phase_arr = []
    
    #Parameters for the number of x and y coordinates
   
    db = lcm_control.DB()
    #Parameter for the number of distinct voltages to be measered at
    volts = 7
    
    #Create empty arrays to store all of the measurements to be taken organized by x index, y index,  and voltage bias
    comp_arr = np.empty((volts, 5000), dtype = np.complex128)
    amp_arr =  np.empty((volts, 5000))
    phase_arr =  np.empty((volts, 5000))
    
    # Initialize VNA Scan Parmeters
    VNATest.init(str(24e9), str(34e9), str(5000))
    
    #Start for loop for the number of voltages given by the parameter 'volts' defined on line 396
    for i in range(volts):
        value = (i-1)*2.0
        if i == 0:
            value = 0
        voltages = lcm_control.make_steering_array(np.full((24, 8), value), 'lb')
        db.steer(voltages)
        print(f"Voltage {value}")
        time.sleep(60)
            
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
            experiment_dir / '2026-03-27-ID2-48inOvertheAirHB.npz',  # "results_phaseStep_40in.npz",
            comp=comp_arr,
            amplitudes=amp_arr,
            phases=phase_arr,
            iteration = i)
        time.sleep(1)

    if value >= 10:
        voltages = lcm_control.make_steering_array(np.full((24, 8), 0.0), 'lb')
        db.steer(voltages)
        time.sleep(10)
        print("reset to zero")
    db.shutdown()
    
if __name__ == "__main__":
    main()