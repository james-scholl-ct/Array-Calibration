# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:57:12 2026

@author: SchollJamesAC3CARILL
"""

import pandas as pd
import re
import numpy as np
import time
import struct
import serial

class USBController:
    def __init__(
            self,
            excel_file: str,
            port: str,
            baudrate: int = 115200,
            start_flag: int = 0xA55A,
            end_flag: int = 0x5AA5,
            timeout: float = 1,
            step_size_v: float = 10/4096,
            ):
        self.dac_max_bits = 4095
        self.start_flag = start_flag
        self.end_flag = end_flag
        self.step_size_v = step_size_v
        self.map_lb, self.map_hb = self.create_dac_map(excel_file)
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            )
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
    @staticmethod
    def crc(data: bytes):
        return sum(data) & 0xFF
    def convert_voltage_array(self, payload):
        payload = np.array(payload)
        payload = np.round(payload/self.step_size_v).astype(int)
        #check if low or high band
        if (payload.shape[0] == 32):
            
        elif (payload.shape[0] == 64):
            
        else:
            raise ValueError("Number of rows should be 32 or 64")
    def build_packet(self, payload:bytes):
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("Payload must be bytes-like")
        payload = bytes(payload)
        if len(payload) > 0xFFFF:
            raise ValueError("Payload too long for 16-bit length")
        header = struct.pack("<HH", self.start_flag, len(payload))
        crc = self.crc(payload)
        trailer = struct.pack("<BH", crc, self.end_flag)
        return header+payload+trailer
    def send(self, payload: bytes):
        array = convert_voltage_array(payload)
        pkt = self.build_packet(array)
        self.ser.write(pkt)
        self.ser.flush()
        
def create_dac_map(self, excel_file):
    dac_map_lb = np.empty((32,32), dtype=object)
    dac_map_hb = np.empty((64,32), dtype=object)
    df = pd.read_excel(excel_file).astype(str).fillna("")
    pattern_lb = re.compile(r"E\d+_\d+")
    pattern_hb = re.compile(r"H\d+_\d+")
    for i, row in df.iterrows():
        for j, value in enumerate(row):
            if pattern_lb.fullmatch(value):
                parts = value[1:].split('_')
                row_idx = int(parts[0]) - 1  # E1 -> index 0
                col_idx = int(parts[1]) - 1  # _2 -> index 1
                # Grab the 3 data points
                data_points = row[j+1:j+3].tolist()
                data_points.insert(0, value)
                # Place into the exact coordinate
                dac_map_lb[row_idx][col_idx] = data_points
            if pattern_hb.fullmatch(value):
                parts = value[1:].split('_')
                row_idx = int(parts[0]) - 1  # H1 -> index 0
                col_idx = int(parts[1]) - 1  # _2 -> index 1
                # Grab the 3 data points
                data_points = row[j+1:j+3].tolist()
                data_points.insert(0, value)
                # Place into the exact coordinate
                dac_map_hb[row_idx][col_idx] = data_points
    return dac_map_lb, dac_map_hb



data0, data1 = create_dac_map(r"C:\Users\SchollJamesAC3CARILL\OneDrive - Carillon Technologies\Documents\2026-01-21 32X32 PINOUT-Fei Controller_local.xlsx")