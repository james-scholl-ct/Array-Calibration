# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:30:07 2026

@author: SpecVision
"""

import numpy as np
from zaber_motion import Units
from zaber_motion.ascii import Connection
import pyvisa
import time
import matplotlib.pyplot as plt

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
                self.instr.control_ren(6) # go to local mode
                self.instr.close()
            except:
                pass
            self.instr = None
    def setup_single_sweep(self, start, stop, points):
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
    def run_and_read_one_sweep(self):
        t0 = time.perf_counter()

        self.instr.write(":INIT1:IMM")             # start single sweep
        self.instr.query("*OPC?")                  # blocks until operation complete

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
        
        t1 = time.perf_counter()
        return t1 - t0, 20*np.log10(np.abs(sdata[:, 0] + 1j * sdata[:, 1]))
    
    def setup_cw_time_sweep(self,FREQ_HZ):
        self.POINTS = 1000
        self.IFBW_HZ =100
        self.SAMPLE_OVERHEAD = .1111111e-3 #ms
        self.instr.write("*CLS")
        self.instr.write(":SENS1:SWE:CW 1") #Turn on CW mode for CH1
        self.instr.write(":SENS1:SWE:TIM:STAT 1") #Sets sweep time state for CH1 on
        self.instr.write(f":SENS1:FREQ:CW {FREQ_HZ}")
        self.instr.write(f":SENS1:BAND {self.IFBW_HZ}")
        self.instr.write(f":SENS1:SWE:CW:POIN {self.POINTS}")
        self.instr.write(":SENS1:HOLD:FUNC HOLD") #Sweep on CH1 is stopped
        self.instr.write(":SENS1:SWE:TIM:DISP 1")
        #self.instr.write(":INIT1:CONT OFF") 
        #Choose measurement, example S21
        self.instr.write(":CALC1:PAR1:DEF S21")
        #Selects trace 1 as active trace
        self.instr.write(":CALC1:PAR1:SEL")
    
        #Binary data is faster than ASCII
        self.instr.write(":FORM:DATA REAL,64")
        self.instr.write(":FORM:BORD SWAP")
        
        self.machine_sweep_time = float(self.instr.query(":SENS1:SWE:TIM?"))
        #print(self.instr.query(":SENS1:SWE:TIM:TYP?"))
        
        self.sample_interval = (1 / self.IFBW_HZ) + self.SAMPLE_OVERHEAD
        self.sweep_time = self.sample_interval * self.POINTS
        self.sample_rate = 1.0 / self.sample_interval      # Hz
        
        
        self.time_axis = np.linspace(0, self.sweep_time, self.POINTS)  # seconds

        print(f"Machine sweep time:      {self.machine_sweep_time:.3f} s")
        print(f"Sweep time:      {self.sweep_time:.3f} s")
        print(f"Sample interval: {self.sample_interval*1000:.4f} ms")
        print(f"Sample rate:     {self.sample_rate:.1f} Hz")

    def start_running(self):
        #self.instr.write(":INIT1:CONT ON")
        #self.instr.write(":INIT:IMM")
        self.instr.write(":TRIG:SEQ:IMM:REM") #Triggers a single sweep, allows command execution during the sweep
        

    def stop_and_read_complex_data(self):
        # Stop continuous sweeping / hold current trace
       
        #Hold sweep on channel 1 
        self.instr.write(":SENS1:HOLD:FUNC HOLD")
        time.sleep(0.2)
    
        # Read complex real/imaginary data
        raw = self.instr.query_binary_values(
            ":CALC1:DATA:SDATA?",
            datatype="d",
            is_big_endian=False,
            container=np.array
        )
        print(f"Got {len(raw)} values, expected {self.POINTS *2}")
        data = np.array(raw).reshape(-1, 2)
        complex_s = data[:, 0] + 1j * data[:, 1]
        
        return 20*np.log10(np.abs(complex_s)), self.sample_interval
    


class Zaber:
    def __init__(self, port):
        self.port = port
        self.max_angle_deg = 70 #placeholder 
        self.min_step_size_deg = .1 #placeholder
        self.speed = 20 #deg/s
        self.accel = 200 #deg/s/s
        self._connect()
        self._init_zaber()
    def _connect(self):
        self.connection = Connection.open_serial_port(self.port)
        device_list = self.connection.detect_devices()
        print("Found {} devices".format(len(device_list)))
        self.device = device_list[0]
        self.axis = self.device.get_axis(1)
    def disconnect(self):
        self.axis.home()
        self.connection.close()
    def _init_zaber(self):
        #if not self.axis.is_homed():
        self.device.settings.set("system.access", 2, Units.NATIVE)

        self.axis.settings.set("limit.max", 180, Units.ANGLE_DEGREES)
        self.axis.settings.set("limit.min", -180, Units.ANGLE_DEGREES)
        
        self.axis.settings.set("accel", self.accel, Units.ANGULAR_ACCELERATION_DEGREES_PER_SECOND_SQUARED)
        self.axis.settings.set("maxspeed", self.speed, Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
        self.axis.settings.set("limit.approach.maxspeed", self.speed/2, Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
        self.axis.home()
        #self.move_abs(90)
    
    def move_abs(self, move_deg, wait_until_idle):
        self.axis.move_absolute(move_deg, Units.ANGLE_DEGREES, wait_until_idle)
    def move_rel(self, move_deg, wait_until_idle):
        self.axis.move_relative(move_deg, Units.ANGLE_DEGREES, wait_until_idle)
    def get_angles(self, span_deg, center_deg, num_points):
        user_max_angle = span_deg/2 + np.abs(center_deg)
        user_step_size = span_deg/num_points
        if user_max_angle > self.max_angle_deg:
            raise ValueError(f"Max angle must be less than +/- {self.max_angle_deg}°, current={user_max_angle}°")
        if user_step_size < self.min_step_size_deg:
            raise ValueError(f"Step size must be >= {self.min_step_size_deg}°, current = {user_step_size}°")
        start_angle_deg = (-span_deg/2)+center_deg
        stop_angle_deg = (span_deg/2)+center_deg
        return start_angle_deg, stop_angle_deg

    def run_scan(self, vna, span_deg, center_deg, num_angle_points, start_freq_ghz, stop_freq_ghz, num_freq_points): 
        magnitude = []
        start_angle_deg, stop_angle_deg = self.get_angles(span_deg, center_deg, num_angle_points)
        half_span = span_deg/2
        offset_angle_deg = (self.speed ** 2) / (2*self.accel) #angle offset to start and stop scan at so that the scan is at constant velocity over the span
        start_angle_real_deg = start_angle_deg - offset_angle_deg -1 #add 1 degree buffer
        stop_angle_real_deg = stop_angle_deg + offset_angle_deg +1 #add 1 degree buffer
        
        start_time_s = self.speed/self.accel 
        scan_time_s = span_deg/self.speed
        
        vna.setup_cw_time_sweep(start_freq_ghz)
        
        #move to start point and wait
        self.move_abs(start_angle_real_deg, wait_until_idle = True)
        
        #start moving to stop point
        self.move_abs(stop_angle_real_deg, wait_until_idle = False)

        #Wait until scanner reaches start of scan position
        while True:
            start_position = self.axis.get_position(Units.ANGLE_DEGREES)
            if start_position >= start_angle_deg:
                start = time.perf_counter()
                break
    
        #start the vna measurment
        vna.start_running()
        #print(start_position)
        
        #wait until scan is complete
        while True:
            stop_position = self.axis.get_position(Units.ANGLE_DEGREES)
            if stop_position >= stop_angle_deg:
                print(stop_position)
                stop = time.perf_counter()
                break
        python_time = stop-start
        print(f"Scan time={python_time}s")
        magnitude, sample_interval = vna.stop_and_read_complex_data()
        #magnitude =1
        #time.sleep(10)
        
        time_axis =  np.linspace(0, len(magnitude) * sample_interval, len(magnitude))
        position_axis = time_axis*self.speed + start_position
        position_axis = position_axis[position_axis < stop_position]
        magnitude = magnitude[0:len(position_axis)]
        #magnitude = np.flip(magnitude)
        self.axis.home()
        return magnitude, position_axis
        

       
        

def main():
    port = "COM4"
    vna_ip_addr = "TCPIP0::192.168.6.150::inst0::INSTR"
    span_deg = 120
    num_angle_points = 20
    center_deg = 0
    start_freq_ghz = 19.3e9
    stop_freq_ghz = 20e9
    num_freq_points = 1
    

    zaber = None
    vna = None
    try:
        vna = VnaInstance(vna_ip_addr)
        #vna=1
        vna.connect()  
        zaber = Zaber(port)
        magnitude, position_axis = zaber.run_scan(vna, span_deg, center_deg, num_angle_points, start_freq_ghz, stop_freq_ghz, num_freq_points)
        print(magnitude)
        print(len(magnitude))
        plt.figure()
        plt.plot(position_axis, magnitude)
        plt.xlabel("Azimuth (°)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"Magnitude Vs Azimuth at {start_freq_ghz/1e9} Ghz")
        plt.ylim(-80, -20)
        plt.show()
    finally:
        if vna is not None:
             vna.disconnect()
        if zaber is not None:
            zaber.disconnect()
    
if __name__ == "__main__":
    main()
    