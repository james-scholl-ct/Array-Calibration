# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:40:26 2026

@author: uconn
"""

from Shared import lcm_control
import numpy as np
import time

db = lcm_control.DB()
voltages = lcm_control.make_element_driving_array(np.full(511,3))
print(voltages)
db.steer(voltages, 'rx')


# try:
#     main()
# except KeyboardInterrupt:
#     pass

