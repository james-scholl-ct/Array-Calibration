# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:00:56 2021

@author: rdpatton
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_laser_scattering():
    base_path = (
        'C:\\Users\\rdpatton\\Documents\\Carillon\\Characterization Testing'
        '\\Pre-Run - 905nm\\Laser Scattering\\'
    )
    file_index = pd.read_excel(base_path + 'Filename Index.xlsx')
    
    fig, axes = plt.subplots(5, dpi=300)
    axis_index = 0
    all_powers = {}

    for filename, sn, angle, elapsed, span in zip(file_index['Filename'].values,
                                                  file_index['SN'].values,
                                                  file_index['Angle'].values,
                                                  file_index['Time'].values,
                                                  file_index['Range'].values):
        data = pd.read_csv(base_path + filename + '.csv', skiprows=3)
        powers = []
        timestamps = []
        encoded_angles = []
        conversion_factor = span / elapsed
        for point in data.values:
            power, timestamp = point.item().split('\t')
            powers.append(float(power))
            timestamps.append(float(timestamp) / 1000)

        timestamps = np.array(timestamps)
        powers = np.array(powers)
        encoded_angles = np.array(timestamps) * conversion_factor - 59
        powers_dB = 10 * np.log10(powers / np.max(powers[:-20]))

        ax = axes[axis_index]
        ax.plot(encoded_angles, powers_dB, label=f'SN00{sn}')
        ax.set_xticks([])
        ax.set_yticks([+10, +5, 0, -5, -10, -15, -20, -25])
        ax.set_yticklabels([+10, +5, 0, -5, -10, -15, -20, -25], fontsize=7)
        ax.set_ylabel(f'Steered @ {angle}°\nSignal (dB)')
        ax.yaxis.label.set_fontsize(10)
        ax.grid(b=True, which='major', axis='y')
        ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        ax.axhline(y=-10, linestyle='-.', color='black', linewidth=1, alpha=0.5)
        plt.subplots_adjust(wspace=0, hspace=0.1)
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        axis_index = (axis_index + 1) % 5
        all_powers[f'{sn}_{angle}'] = (encoded_angles, powers_dB)
        
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    axes[0].set_title('Laser Scattering', fontsize=20)
        
    axes[-1].set_xticks([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70])
    axes[-1].set_xticklabels([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70],
                             fontsize=10)
    axes[-1].set_xlabel('Observation Angle (°), relative to chip normal')
    axes[-1].xaxis.label.set_fontsize(15)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.62, 1),
                   ncol=4, fancybox=True, shadow=True, prop=dict(size=10))
    
    plt.show()
    plt.pause(5)
    plt.tight_layout()
    
    return all_powers


def plot_broadband_scattering():
    base_path = (
        'C:\\Users\\rdpatton\\Documents\\Carillon\\Characterization Testing'
        '\\Pre-Run - 905nm\\Broadband Scattering\\'
    )
    file_index = pd.read_excel(base_path + 'Filename Index.xlsx')
    
    fig, axes = plt.subplots(5, dpi=300)
    axis_index = 0

    for filename, sn, angle, elapsed, span in zip(file_index['Filename'].values,
                                                  file_index['SN'].values,
                                                  file_index['Angle'].values,
                                                  file_index['Time'].values,
                                                  file_index['Range'].values):
        data = pd.read_csv(base_path + filename + '.csv', skiprows=3)
        powers = []
        timestamps = []
        encoded_angles = []
        conversion_factor = span / elapsed
        for point in data.values:
            power, timestamp = point.item().split('\t')
            powers.append(float(power))
            timestamps.append(float(timestamp) / 1000)

        timestamps = np.array(timestamps)
        powers = np.array(powers)
        encoded_angles = np.array(timestamps) * conversion_factor - 60
        powers_dB = 10 * np.log10(powers / np.max(powers))

        ax = axes[axis_index]
        ax.plot(encoded_angles, powers_dB, label=f'SN00{sn}')
        ax.set_xticks([])
        ax.set_yticks([+5, 0, -5, -10, -15, -20, -25, -30])
        ax.set_yticklabels([+5, 0, -5, -10, -15, -20, -25, -30],
                           fontsize=7)
        ax.set_ylabel(f'Steered @ {angle}°\nSignal (dB)')
        ax.yaxis.label.set_fontsize(10)
        ax.grid(b=True, which='major', axis='y')
        ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        ax.axhline(y=-10, linestyle='-.', color='black', linewidth=1, alpha=0.5)
        plt.subplots_adjust(wspace=0, hspace=0.1)
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        axis_index = (axis_index + 1) % 5
        
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    axes[0].set_title('Broadband Source (T ~ 2800K) Scattering - Measured at 905nm',
                      fontsize=20)
        
    axes[-1].set_xticks([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70])
    axes[-1].set_xticklabels([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70],
                             fontsize=10)
    axes[-1].set_xlabel('Observation Angle (°), relative to chip normal')
    axes[-1].xaxis.label.set_fontsize(15)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1),
                   ncol=4, fancybox=True, shadow=True, prop=dict(size=10))
    
    plt.show()
    plt.pause(5)
    plt.tight_layout()
    
    return powers_dB