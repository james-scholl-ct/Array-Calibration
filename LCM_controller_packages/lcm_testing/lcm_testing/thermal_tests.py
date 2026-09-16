# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:51:49 2021

@author: rdpatton
"""

from matplotlib import pyplot as plt

import lcm_testing.strehl_ratio as sr

def compare_thermals(sn, base_path, bucket_dict=None):
    """
    Do the steering and flatness Strehls for both the 6kOhm data and 3kOhm data.

    Parameters
    ----------
    sn : int
        The serial number on which the thermal testing was performed.
    base_path : str
        This should be the path that the Wavefront Data lives in. The sub_paths
        defined herein navigate to the Thermal and SO/CF subfolders.
    bucket_dict : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    sn_list = [(sn, '6k\\'), (sn, '3k\\')]
    
    # Steering Strehl Ratio
    sub_path = base_path + 'Thermal\\Steering Order Strehl Ratio\\'
    if bucket_dict is None:
        bucket_dict = sr.build_steering_bucket_dict()

    steering_results = sr.plot_all_steering_strehls(sn_list, sub_path,
                                                      bucket_dict)

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    
    # Plot original data
    sub_path = base_path + 'Steering Order Strehl Ratio\\'
    sr.plot_all_steering_strehls([sn], sub_path, bucket_dict=bucket_dict,
                                 fig1=fig1, fig2=fig2, linestyle='dotted')
    
    fig1.suptitle('Thermal Comparison')
    axes1 = fig1.axes
    for ax in axes1:
        ax.grid()
    axes2 = fig2.axes
    axes2[0].set_title('Steering Order Strehl Ratio, Thermal Comparison')
    axes2[0].grid()
    
    # Flatness Strehl Ratio
    sub_path = base_path + 'Thermal\\Chip Flatness Strehl Ratio\\'
    flatness_result = sr.plot_all_flatness_strehls(sn_list, sub_path)
    fig3 = plt.figure(3)
    
    # Plot original data
    sub_path = base_path + 'Chip Flatness Strehl Ratio\\'
    sr.plot_all_flatness_strehls([sn], sub_path, fig=fig3)
    axes3 = fig3.axes
    axes3[0].set_title('Chip Flatness Strehl Ratio - Reference = Mirror, Thermal Comparison')
    
    return steering_results, flatness_result
