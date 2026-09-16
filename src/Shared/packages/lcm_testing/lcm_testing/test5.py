# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:07:00 2021

@author: rdpatton
"""

import time
import numpy as np
from matplotlib import pyplot as plt

import thorlabs_kinesis as tk
import lcm_control as lc


def main():
    db = lc.DB(-70)
    pa = tk.PositionAlignerWrapper()
    pa.connect()
    pa.set_mode(tk.MONITOR)
    xs = []; ys = []; ss = []; ts=[]
    pa.stop_polling()
    pa.start_polling(1)
    db.steer(1)
    time.sleep(5)
    start = time.time()
    for i in range(100000):
        x, y, s = pa.get_position()
        xs.append(x)
        ys.append(y)
        ss.append(s)
        ts.append(time.time() - start)
    s1a = time.time() - start
    db.steer(20)
    s1b = time.time() - start
    for i in range(100000):
        x, y, s = pa.get_position()
        xs.append(x)
        ys.append(y)
        ss.append(s)
        ts.append(time.time() - start)
    s2a = time.time() - start
    db.steer(19)
    s2b = time.time() - start
    for i in range(100000):
        x, y, s = pa.get_position()
        xs.append(x)
        ys.append(y)
        ss.append(s)
        ts.append(time.time() - start)
    s3a = time.time() - start
    db.steer(20)
    s3b = time.time() - start
    for i in range(100000):
        x, y, s = pa.get_position()
        xs.append(x)
        ys.append(y)
        ss.append(s)
        ts.append(time.time() - start)
    end = time.time()
    db.standby()
    print(f'Time Elapsed: {end - start:.3f}')
    
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ss = np.asarray(ss)
    
    fig, axes = plt.subplots(2)
    axes[0].axvline(s1a, color='r')
    axes[0].axvline(s2a, color='r')
    axes[0].axvline(s3a, color='r')
    axes[0].axvline(s1b, color='limegreen')
    axes[0].axvline(s2b, color='limegreen')
    axes[0].axvline(s3b, color='limegreen')
    axes[0].plot(ts, xs, label='XDIFF')
    axes[0].plot(ts, ys, label='YDIFF')
    axes[0].plot(ts, ss, label='SUM')
    axes[0].legend()
    axes[0].set_xlabel('Time Elapsed (s)')
    axes[0].set_ylabel('Raw Output (V)')
    axes[0].set_title('Switching Test - order 1 to 20 to 19 to 20\nPDQ30C Centered on 20')
    
    axes[1].axvline(s1a, color='r')
    axes[1].axvline(s2a, color='r')
    axes[1].axvline(s3a, color='r')
    axes[1].axvline(s1b, color='limegreen')
    axes[1].axvline(s2b, color='limegreen')
    axes[1].axvline(s3b, color='limegreen')
    axes[1].plot(ts, np.where(xs/ss < 10, np.where(xs/ss > -10, xs/ss, 0), 0), label='XDIFF')
    axes[1].plot(ts, np.where(ys/ss < 10, np.where(ys/ss > -10, ys/ss, 0), 0), label='YDIFF')
    axes[1].plot(ts, ss, label='SUM')
    axes[1].legend()
    axes[1].set_xlabel('Time Elapsed (s)')
    axes[1].set_ylabel('Normalized Output')
    axes[1].set_title('Switching Test - order 1 to 20 to 19 to 20\nPDQ30C Centered on 20')