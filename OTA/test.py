# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:40:26 2026

@author: uconn
"""

from reflectek_tools import lcm_control
import numpy as np
import time


def main():
    
    db = lcm_control.DB()
    # steerPattern = np.loadtxt(r"C:\Users\uconn\Downloads\MiliBoxRangeOptimizations\LCD_Controller\19.0GHz\CMA\0_0_2026_9_18\output\sweep_20260929_125056\theta0_phi0\applied_voltages.csv", delimiter=",") 
    # voltages = lcm_control.make_steering_array(steerPattern, 'lb')
    voltages = lcm_control.make_element_driving_array(np.full(511,9,dtype=float))
    db.steer(voltages, 'rx')
    while True:
        time.sleep(1)

try:
    main()
except KeyboardInterrupt:
    pass

