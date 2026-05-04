# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 11:22:50 2025

@author: SchollJamesAC3CARILL
"""
import pyvisa
import numpy as np

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
        meas = "S21"
    
        self.instr.write("SYST:PRES")
        self.instr.write("*CLS")
    
        #self.instr.write("DISP:WIND1:STAT ON")
        self.instr.write(f"CALC1:PAR:DEF:EXT 'Meas1',{meas}")
        #self.instr.write("DISP:WIND1:TRAC1:FEED 'Meas1'")
        self.instr.write("CALC1:PAR:SEL 'Meas1'")
    
        self.instr.write("SENS1:SWE:TYPE LIN")
        self.instr.write(f"SENS1:FREQ:STAR {start}")
        self.instr.write(f"SENS1:FREQ:STOP {stop}")
        self.instr.write(f"SENS1:SWE:POIN {points}")
    
        self.instr.write("INIT1:CONT OFF")
        self.instr.write("SENS1:SWE:MODE SING")
    
        self.instr.write("FORM:DATA REAL,64")
        self.instr.write("FORM:BORD SWAP")
    
        print("Sweep type:", self.instr.query("SENS1:SWE:TYPE?").strip())
        print("Start Hz  :", self.instr.query("SENS1:FREQ:STAR?").strip())
        print("Stop Hz   :", self.instr.query("SENS1:FREQ:STOP?").strip())
        print("Points    :", self.instr.query("SENS1:SWE:POIN?").strip())
        print("Selected  :", self.instr.query("CALC1:PAR:SEL?").strip())
        print("Error     :", self.instr.query("SYST:ERR?").strip())
    
        # actually trigger a measurement
        self.instr.write("INIT1:IMM")
        self.instr.query("*OPC?")
    
        raw = self.instr.query_binary_values(
            "CALC1:DATA? SDATA",
            datatype="d",
            is_big_endian=False,
            container=np.array,
        )
    
        print("raw size:", raw.size)
    
        if raw.size % 2 != 0:
            raise RuntimeError(f"Unexpected raw data length: {raw.size}")
    
        sdata = raw.reshape((-1, 2))
        return sdata[:, 0] + 1j * sdata[:, 1]