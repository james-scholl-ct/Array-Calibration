# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 16:24:37 2021

@author: rdpatton

https://f.hubspotusercontent40.net/hubfs/9035299/Documents/AAS-913-318C-Temperature-resistance-curves-071816-web.pdf

This calculator applies to Amphenol NTC Thermistors, Material Type F only!

Ranges in the coefficient dictionaries are as follows for R_t / R-25:
    68.600 to 3.274 => 1
    3.274 to 0.36036 => 2
    0.36036 to 0.06831 => 3
    0.06831 to 0.01872 => 4
"""

import numpy as np


def calc_temp(Rt, R25=10e3):
    A = {1: 3.3538646e-3,
         2: 3.3540154e-3,
         3: 3.3539264e-3,
         4: 3.3368620e-3}

    B = {1: 2.5654090e-4,
         2: 2.5627725e-4,
         3: 2.5609446e-4,
         4: 2.4057263e-4}

    C = {1: 1.9243889e-6,
         2: 2.0829210e-6,
         3: 1.9621987e-6,
         4: -2.6687093e-6}

    D = {1: 1.0969244e-7,
         2: 7.3003206e-8,
         3: 4.6045930e-8,
         4: -4.0719355e-7}

    ratio = Rt / R25
    range_index = calc_resistance_range(ratio)
    a = A[range_index]
    b = B[range_index]
    c = C[range_index]
    d = D[range_index]

    inv_t = a + b * np.log(ratio) + c * np.log(ratio)**2 + d * np.log(ratio)**3

    T = 1 / inv_t - 273.15
    return T


def calc_resistance(T_C, R25=10e3):
    A = {1: -1.4122478e1,
         2: -1.4141963e1,
         3: -1.4202172e1,
         4: -1.6154078e1}

    B = {1: 4.4136033e3,
         2: 4.4307830e3,
         3: 4.4975256e3,
         4: 6.8483992e3}

    C = {1: -2.9034189e4,
         2: -3.4078983e4,
         3: -5.8421357e4,
         4: -1.0004049e6}

    D = {1: -9.3875035e6,
         2: -8.8941929e6,
         3: -5.9658796e6,
         4: 1.1961431e8}
    
    T_K = T_C + 273.15

    range_index = calc_temp_range(T_C)
    a = A[range_index]
    b = B[range_index]
    c = C[range_index]
    d = D[range_index]
    
    ratio = np.exp(a + b/T_K + c/T_K**2 + d/T_K**3)
    
    return ratio * R25

    
def calc_temp_range(ratio):
    if ratio > 68.6:
        raise ValueError('Resistance value exceeds usable range.')
    elif ratio >= 3.274:
        return 1
    elif ratio >= 0.36036:
        return 2
    elif ratio >= 0.06831:
        return 3
    elif ratio >= 0.01872:
        return 4
    else:
        raise ValueError('Resistance value excceds usable range.')


def calc_resistance_range(ratio):
    if ratio > 68.6:
        raise ValueError('Resistance value exceeds usable range.')
    elif ratio >= 3.274:
        return 1
    elif ratio >= 0.36036:
        return 2
    elif ratio >= 0.06831:
        return 3
    elif ratio >= 0.01872:
        return 4
    else:
        raise ValueError('Resistance value excceds usable range.')
