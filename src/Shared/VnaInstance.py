# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 11:22:50 2025

@author: SchollJamesAC3CARILL
"""
import pyvisa

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