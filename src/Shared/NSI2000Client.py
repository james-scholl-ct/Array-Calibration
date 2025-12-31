# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 11:17:10 2025

@author: SchollJamesAC3CARILL
"""
import win32com.client
import gc
import numpy as np
from pathlib import Path
import time

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
        
    def run_scan_get_hor_amp(self, filename, beam):
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
        self.cmd.SELECT_BEAM(beam)
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