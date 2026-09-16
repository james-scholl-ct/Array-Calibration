# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:01:08 2021

@author: rdpatton

Algorithm (at each steering angle):
    >>> Steering Order Strehl Ratio <<<
    Use Function: plot_all_steering_strehls
    1. Model the diffraction-limited spot profile for the geometeries
       applicable to that steering angle.
    2. Take cross-sections in X and Y along the major and minor axes.
    3. Iterate to determine the bucket size, in each axis separately, which
       contains 1-1/e^2 of the total energy.
    4. Position the bucket under the real data from that steering angle, such
       that it contains the most amount of energy for that bucket.
    5. Integrate with the bucket at that position.
    6. Repeat steps 1-5 for specular reflection.
    7. Correct the steered bucket energy for ratio of total scene energy in
       steered frame relative to specular reflection.
    8. The ratio from corrected steered bucket energy to specular bucket
       energy is the effective Steering Strehl Ratio in X and Y. The product
       of those two ratios is the complete Effective Steering Wavefront
       Efficiency.

    >>> Chip Flatness Strehl Ratio <<<
    Use Function: plot_all_flatness_strehls
    9. Now take the 10° specular data.
    10. Measure beam width.
    11. Model the diffraction-limited spot profile for the geometries at 10°.
    12. Take cross-sections in X and Y along the major and minor axes.
    13. Iterate to determine the bucket size, in each axis separately, which
        contains 1-1/e^2 of the total energy.
    14. Position the bucket under the 10° mirror specular refleciton data,
        such that it contains the most amount of energy for that bucket.
    15. Integrate with the bucket at that position.
    16. Repeat steps 14-15 for the 10° specular reflection (bucket size will
        remain constant in this case, since the aperture geometry is fixed).
    17. Normalize the integral by the ratio of total energy in each scene.
    18. The resulting ratio from corrected LCM bucket energy to mirror bucket
        energy is the Chip Flatness Strehl Ratio in X and Y. The product of
        those two ratios is the complete Chip Flatness Strehl Ratio.
"""

import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import simps

import lcm_testing.fft_psf as fft_psf


def plot_all_flatness_strehls(sn_list, base_path, beam_dia=11e-3,
                              fig=None, linestyle=None):
    """
    Main entry point for plotting all the Chip Flatness Strehl Ratios computed
    using the energy-in-a-bucket method for all serial numbers in sn_list.
    Plots a bar chart with the result for each serial number in sn_list.

    Parameters
    ----------
    sn_list : list of ints
        The list of SNs whose data we are to plot.
    base_path : str
        The base path to the chip flatness data.
    beam_dia : float, optional
        The incident beam diameter, in meters. The default is 11e-3.

    Returns
    -------
    SRs : list of floats
        The list of computed Chip Flatness Strehl Ratios, 1 for each serial
        number in sn_list.

    """
    # Set up figure
    if fig is None:
        fig = plt.figure(dpi=200)
    
    # Set up base path for data - needs to be changed for each run.
    SRs = []
    
    # Run Chip Flatness Strehl Ratio computation by energy-in-bucket method
    print('\n***Calculating Chip Flatness Strehl Ratio by Bucket Method***')
    for item in sn_list:
        if isinstance(item, tuple):
            sn, extension = item
        else:
            sn = item
            extension = ''
        path = base_path + extension
        SR = calc_flatness_strehl(sn, path, beam_dia)
        SRs.append(SR[0] * SR[1])

    # Fix color map and open figure
    colors = plt.get_cmap('tab10')(np.arange(len(SRs)))
    
    # Plot the SR results
    for item, SR, c in zip(sn_list, SRs, colors):
        if isinstance(item, tuple):
            sn, extension = item
            note = f' @ {extension[:-1]}Ohms'
        else:
            sn = item
            extension = ''
            note = ''
        plt.bar(f'SN00{sn}{note}', SR, zorder=3, alpha=0.9)
    plt.ylim([0, 1])
    plt.grid(axis='y', zorder=0, alpha=0.5)
    plt.ylabel('Strehl Ratio')
    plt.xlabel('Serial Number')
    plt.title('Chip Flatness Strehl Ratio - Reference = Mirror')
    
    # Annotate SR value above plotted bar
    ax = fig.gca()
    rects = ax.patches
    if len(SRs) == 1 and len(rects) > 1:
        rects = [rects[-1]]
    for rect, SR in zip(rects, SRs):
        height = rect.get_height()
        ax.annotate(f'{SR:.3f}',
                    (rect.get_x() + rect.get_width() / 2, height),
            ha='center', va='bottom')
    plt.show()
    
    return SRs


def plot_all_steering_strehls(sn_list, base_path, bucket_dict=None,
                              fig1=None, fig2=None, linestyle=None):
    """
    Main entry point for plotting all the Steering Order Strehl Ratios
    computed using the energy-in-a-bucket method for all serial numbers in
    sn_list. Plots a curve of SR vs steering angle for each serial number in
    sn_list.
    
    SRs are calculated individually in X and Y, then the combined SR is
    plotted separately as the product of SRx and SRy.
    
    A dictionary of bucket sizes to use can be provided in bucket_dict to
    save on computation time for repeat calls to this function; otherwise, it
    will be built from scratch every time this function is called.

    Parameters
    ----------
    sn_list : list of ints
        The list of SNs whose data we are to plot.
    base_path : str
        The path to look for files.
    bucket_dict : dictionary of (float, float) tuples, optional
        A dictionary containing the x and y bucket sizes to use at each angle.
        Format is {angle: (size_x, size_y)}, the same as returned in
            build_steering_bucket_dict().
        The default is None.

    Returns
    -------
    data : dict of (list of floats, list of floats) tuples
        A dictionary containing the computed Steering Order Strehl Ratios
        in X and Y separately, for each serial number in sn_list. Each list of
        SRxs and SRys is the computed SR at angles from [-60 to +60°].
        Format is {sn: ([SRxs_vs_angle], [SRYs_vs_angle])}.

    """
    data = {}
    
    # Build the dictionary of bucket sizes to use for integrating the data
    if bucket_dict is None:
        bucket_dict = build_steering_bucket_dict()

    angles = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    if fig1 is None:
        fig1, axes1 = plt.subplots(3, dpi=200)
    if fig2 is None:
        fig2, axes2 = plt.subplots(1, dpi=200)
    axes1 = fig1.axes
    axes2 = fig2.axes
    if linestyle is None:
        linestyle = 'solid'
    
    # For each SN, calculate and plot the Steering Order Strehl Ratio
    for item in sn_list:
        if isinstance(item, tuple):
            sn, extension = item
            note = f' @ {extension[:-1]}Ohms'
        else:
            sn = item
            extension = ''
            note = ''
        path = base_path + extension
        SRxs = []
        SRys = []
        for angle in angles:
            srs = calc_steering_strehl(sn, angle, path, bucket_dict)
            SRxs.append(srs[0])
            SRys.append(srs[1])
        axes1[1].plot(angles, SRxs, label=f'SN00{sn}{note}', linestyle=linestyle)
        axes1[2].plot(angles, SRys, label=f'SN00{sn}{note}', linestyle=linestyle)
        axes1[0].plot(angles, np.array(SRxs) * np.array(SRys),
                     label=f'SN00{sn}{note}', linestyle=linestyle)
        data[sn] = (SRxs, SRys)
        axes2[0].plot(angles, np.array(SRxs) * np.array(SRys),
                   label=f'SN00{sn}{note}', linestyle=linestyle)
    
    # Format plots for reporting
    axes1[0].set_title('Combined Steering Order Strehl Ratio')
    axes1[0].set_ylabel('Strehl Ratio')
    axes1[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes1[0].set_ylim([0, 1])
    axes1[0].grid(axis='y')
    axes1[0].legend()
    axes1[1].set_title('Steering Order Strehl Ratio in (major) X-axis')
    axes1[1].set_ylabel('Strehl Ratio')
    axes1[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes1[1].set_ylim([0, 1])
    axes1[1].grid(axis='y')
    axes1[1].legend()
    axes1[2].set_title('Steering Order Strehl Ratio in (minor) Y-axis')
    axes1[2].set_xlabel('Nominal Steering Angle (°)')
    axes1[2].set_ylabel('Strehl Ratio')
    axes1[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes1[2].set_ylim([0, 1])
    axes1[2].grid(axis='y')
    axes1[2].legend()
    
    axes2[0].set_title('Steering Order Strehl Ratio')
    axes2[0].set_xlabel('Nominal Steering Angle (°)')
    axes2[0].set_ylabel('Strehl Ratio')
    axes2[0].legend()
    axes2[0].grid()
    
    return data


def calc_flatness_strehl(sn, base_path, beam_dia, bucket_sizes=None):
    """
    Calculates the Chip Flatness Strehl Ratio in X and Y for the serial number
    provided, with the data at the path location provided. See the algorithm
    in the header of this file for detail on the steps involved. The measured
    data for specular reflection off the mirror is used as the baseline for
    this SR computation.
    
    Returns the computed SRs in X and Y, along with the total energy ratio in
    X and Y, which were used in this function to normalize the bucket energies.

    Parameters
    ----------
    sn : int
        The serial number of interest.
    base_path : str
        The base path to the chip flatness data.
    beam_dia : float, optional
        The incident beam diameter, in meters. The default is 11e-3.
    bucket_sizes : tuple of floats, optional
        A tuple containing the x and y bucket sizes to use at each angle.
        Format is (size_x, size_y).
        The default is None.

    Returns
    -------
    SR_x : float
        The Chip Flatness Strehl Ratio in the x-direction.
    SR_y : float
        The Chip Flatness Strehl Ratio in the y-direction.
    TR_x : float
        The total energy ratio of DUT data to mirror data, in the x-direction.
    TR_y : float
        The total energy ratio of DUT data to mirror data, in the y-direction.

    """
    print(f'\n>>>Computing Chip Flatness Strehl Ratio for SN00{sn}<<<')
    
    # Find bucket sizes for specified beam diameter, if not provided
    if bucket_sizes is None:
        bucket_x, bucket_y = do_flatness_bucket_model(beam_dia)
    else:
        bucket_x, bucket_y = bucket_sizes
    print(f'Bucket Sizes: ({bucket_x:.3f}µm, {bucket_y:.3f}µm)')
    
    # Find bucket energy for mirror (baseline)
    path = base_path + 'mirror*'
    mirror_energy_x, mirror_energy_y, mirror_total_x, mirror_total_y = \
        find_actual_bucket_energy(path, bucket_x, bucket_y)
    
    # Find bucket energy for DUT
    path = base_path + f'SN00{sn}*'
    dut_energy_x, dut_energy_y, dut_total_x, dut_total_y = \
        find_actual_bucket_energy(path, bucket_x, bucket_y)
    
    # The total energy ratio, later used for normalization
    TR_x = dut_total_x / mirror_total_x
    TR_y = dut_total_y / mirror_total_y
    
    print(f'Ratio of Totals: ({TR_x:.3f}, {TR_y:.3f})')
    
    # Calculate (uncorrected) Effective Strehl Ratio
    uSR_x = dut_energy_x / mirror_energy_x
    uSR_y = dut_energy_y / mirror_energy_y
    
    print(f'Pre-Corrected SR: ({uSR_x:.3f}, {uSR_y:.3f})')
    
    # Calculate (corrected) Effective Strehl Ratio
    SR_x = uSR_x / TR_x
    SR_y = uSR_y / TR_y
    
    print(f'Final SR: ({SR_x:.3f}, {SR_y:.3f}) = {SR_x * SR_y:.3f}')
    
    return SR_x, SR_y, TR_x, TR_y


def calc_steering_strehl(sn, steering_angle, base_path, bucket_dict=None):
    """
    Calculates the Steering Order Strehl Ratio in X and Y for the serial number
    provided, with the data at the path location provided. See the algorithm
    in the header of this file for detail on the steps involved. The measured
    data for specular reflection off the HOBS is used as the baseline for
    this SR computation.
    
    Returns the computed SRs in X and Y, along with the total energy ratio in
    X and Y, which were used in this function to normalize the bucket energies.

    Parameters
    ----------
    sn : int
        The serial number of interest.
    steering_angle : int
        The steering angle of interest.
    base_path : str
        The path to look for files.
    bucket_dict : dictionary of (float, float) tuples, optional
        A dictionary containing the x and y bucket sizes to use at each angle.
        Format is {angle: (size_x, size_y)}, the same as returned in
            build_steering_bucket_dict().
        The default is None.

    Returns
    -------
    SR_x : float
        The Chip Flatness Strehl Ratio in the x-direction.
    SR_y : float
        The Chip Flatness Strehl Ratio in the y-direction.
    TR_x : float
        The total energy ratio of DUT data to mirror data, in the x-direction.
    TR_y : float
        The total energy ratio of DUT data to mirror data, in the y-direction.

    """
    print(f'\n*** Calculating Strehl Ratio for SN00{sn}, '
          f'{steering_angle}° ***')

    if bucket_dict is None:
        # Find bucket sizes for steered beam
        steer_bucket_x, steer_bucket_y = do_steering_bucket_model(steering_angle)
        print('Steered Bucket Sizes: '
              f'({steer_bucket_x:.3f}µm, {steer_bucket_y:.3f}µm)')
        # Find bucket sizes for specular beam
        spec_bucket_x, spec_bucket_y = do_steering_bucket_model(70)
        print('Specular Bucket Sizes: '
              f'({spec_bucket_x:.3f}µm, {spec_bucket_y:.3f}µm)')
    else:
        steer_bucket_x, steer_bucket_y = bucket_dict[steering_angle]
        spec_bucket_x, spec_bucket_y = bucket_dict[70]
    
    # Find bucket energies for steered angle
    path = base_path + f'SN00{sn}_{steering_angle}*'
    steer_energy_x, steer_energy_y, steer_total_x, steer_total_y = \
        find_actual_bucket_energy(path, steer_bucket_x, steer_bucket_y)
    
    # Find bucket energies for specular reflection
    path = base_path + f'SN00{sn}_70*'
    spec_energy_x, spec_energy_y, spec_total_x, spec_total_y = \
        find_actual_bucket_energy(path, spec_bucket_x, spec_bucket_y)
    
    TR_x = steer_total_x / spec_total_x
    TR_y = steer_total_y / spec_total_y
    
    print(f'Ratio of Totals: ({TR_x:.3f}, {TR_y:.3f})')
    
    # Calculate (uncorrected) Effective Strehl Ratio
    uSR_x = steer_energy_x / spec_energy_x
    uSR_y = steer_energy_y / spec_energy_y
    
    print(f'Pre-Corrected SR: ({uSR_x:.3f}, {uSR_y:.3f})')
    
    # Calculate (corrected) Effective Strehl Ratio
    SR_x = uSR_x / TR_x
    SR_y = uSR_y / TR_y
    
    print(f'Final SR: ({SR_x:.3f}, {SR_y:.3f}) = {SR_x * SR_y:.3f}')
    
    return SR_x, SR_y, TR_x, TR_y


def do_flatness_bucket_model(beam_dia):
    """
    Runs the FFT model for the underfilled at 10° incident angle, to calculate
    the 1/e^2 bucket (spot) size in both X and Y. These values should be equal
    for this circular aperture being imaged.
    
    Returns the computed bucket sizes in X and Y.

    Parameters
    ----------
    beam_dia : float
        The incident beam diameter, in meters.
        Note: The beam is forelengthened to an ellipse when it hits the chip,
        but since it's undergoing specular reflection, the reflected beam in
        collimated space is circular.

    Returns
    -------
    bucket_x : float
        The 1/e^2 spot diameter along the X-axis
    bucket_y : float
        The 1e^2 spot diameter along the Y-axis

    """
    _, result, _, _ = fft_psf.main(beam_dia, 64e-3,
                                   rho=10, ftheta=-10,
                                   astop_px=200, N=2000, zoom=3,
                                   integrate=False, plot=False,
                                   aperture_func='ellipse')
    
    # Unpack the PSF result
    psf_image = result[1]
    psf_coords = result[0]
    
    # Find the peak indices and take the peak cross-sections
    idx = psf_image.argmax() % psf_image.shape[1]
    idy = psf_image.argmax() // psf_image.shape[1]
    horiz_slice = psf_image[idy]
    vert_slice = [im[idx] for im in psf_image]
    
    # Find the bucket size
    x_a, x_b = find_bucket_limits(horiz_slice, psf_coords)
    y_a, y_b = find_bucket_limits(vert_slice, psf_coords)
    bucket_x = x_b - x_a
    bucket_y = y_b - y_a
    
    return bucket_x, bucket_y


def do_steering_bucket_model(steering_angle):
    """
    Runs the FFT model for the fully-illuminated HOBS aperture, at the
    specified steering angle, to calculate the 1/e^2 bucket (spot) size in
    both X and Y. Unlike the underfilled aperture used in the chip flatness
    model, the X and Y bucket sizes will be unequal.
    
    Returns the computed bucket sizes in X and Y.

    Parameters
    ----------
    steering_angle : int
        The steering angle to model.

    Returns
    -------
    bucket_x : float
        The 1/e^2 spot diameter along the X-axis
    bucket_y : float
        The 1/e^2 spot diameter along the X-axis

    """
    _, result, _, _ = fft_psf.main(27e-3, 64e-3,
                                   rho=steering_angle, ftheta=70,
                                   astop_px=200, N=2000, zoom=3,
                                   integrate=False, plot=False,
                                   aperture_func='rect')
    
    # Unpack the PSF result
    psf_image = result[1]
    psf_coords = result[0]
    
    # Find the peak indices and take the peak cross-sections
    idx = psf_image.argmax() % psf_image.shape[1]
    idy = psf_image.argmax() // psf_image.shape[1]
    horiz_slice = psf_image[idy]
    vert_slice = [im[idx] for im in psf_image]
    
    # Find the bucket size
    x_a, x_b = find_bucket_limits(horiz_slice, psf_coords)
    y_a, y_b = find_bucket_limits(vert_slice, psf_coords)
    bucket_x = x_b - x_a
    bucket_y = y_b - y_a
    
    return bucket_x, bucket_y
    

def find_actual_bucket_energy(path, bucket_x, bucket_y):
    """
    Finds and returns the maximum possible bucket energy for the data file at
    the specified path. Bucket sizes are provided, but their best positions
    are unknown. This function loads the data, then calls a helper function to
    scan the bucket along the data curve, to find the position that returns
    the maximum possible energy within a bucket of the specified size.

    Parameters
    ----------
    path : str
        The path to the data file.
    bucket_x : float
        The width of the bucket to use for data on the x-axis.
    bucket_y : float
        The width of the bucket to use for data on the y-axis.

    Returns
    -------
    bucket_energy_x : float
        Energy in the bucket along the x-axis.
    bucket_energy_y : float
        Energy in the bucket along the y-axis.
    total_x : float
        Total energy along the x-axis.
    total_y : float
        Total energy along the y-axis.

    """
    # Load datafile at steering angle
    for f in glob.glob(path):
        pdata = pd.read_csv(f, usecols=[0, 1, 2, 3], skiprows=55,
                            encoding='latin-1')
    
        X = np.asarray(pdata[r'Position X ROI#1'].dropna())
        Y = np.asarray(pdata[r'Position Y ROI#1'].dropna())
        xi = np.asarray(pdata[r' Intensity X ROI#1'].dropna())
        yi = np.asarray(pdata[r' Intensity Y ROI#1'].dropna())
 
    # Find max bucket energies for X and Y
    bucket_energy_x = optimal_bucket_scan(xi, X, bucket_x)
    bucket_energy_y = optimal_bucket_scan(yi, Y, bucket_y)
    
    # Find total integral for X and Y
    total_x = simps(xi, X)
    total_y = simps(yi, Y)
    
    return bucket_energy_x, bucket_energy_y, total_x, total_y


def optimal_bucket_scan(curve, coords, bucket_size):
    """
    Scans a bucket of the given size along the entire provided curve,
    integrating at each position to find the maximum possible integral. Starts
    with the bucket at the far left of the curve, and scans to the right by
    incrementing the bucket's left index one at a time.
    
    A helper function is used to determine the index of the bucket's right
    index which puts it as close as possible to the specified bucket size.

    Parameters
    ----------
    curve : ndarray of floats
        The intensity curve to integrate.
    coords : ndarray of floats
        The coordinates of the intensity curve
    bucket_size : float
        The desired width of the bucket to integrate. Generally, the
        coordinates will not exactly match this bucket size, so the function
        will find an integration interval that matches this bucket size as
        closely as possible.

    Returns
    -------
    max_bucket_energy : float
        The maximum value of the integral as the integration interval (bucket)
        is scanned along the entire curve.

    """
    # Initialize tracking values
    start_index = 0
    end_index = find_nearest(coords, bucket_size)
    max_bucket_energy = 0
    
    
    # Scan across entire axis, finding the max energy that falls into bucket
    while coords[start_index] + bucket_size < coords[-1]:
        bucket_energy = simps(curve[start_index:end_index+1],
                              coords[start_index:end_index+1])
        if bucket_energy > max_bucket_energy:
            max_bucket_energy = bucket_energy
        
        start_index += 1
        end_index = find_nearest(coords, coords[start_index] + bucket_size)
    
    return max_bucket_energy


def find_nearest(array, value):
    """
    Finds the index of the nearest value in the specified array to the
    specified value.

    Parameters
    ----------
    array : ndarray of floats
        The array to search.
    value : float
        The value to match.

    Returns
    -------
    index : int
        The index of the value in the array which comes closest to matching
        the specified value.

    """
    index = (np.abs(array - value)).argmin()
    return index


def find_bucket_limits(intensity, coords, goal=1-1/np.e**2):
    """
    Finds the smallest integration interval (bucket) which encapsulates
    1-1/e^2 of the total energy in the intensity curve. This can also be
    described as a spot size. This function returns the lower and upper limits
    of integration rather than simply the bucket size.
    
    For all use cases, the lower and upper limits should be symmetric about
    the y-axis (or off-by-one), unless wavefront errors are specified. If this
    happens, this function's algorithm will likely need to be modified to deal
    with asymmetric intensity.

    Parameters
    ----------
    intensity : ndarray of floats
        The intensity curve to integrate.
    coords : ndarray of floats
        The coordinates of the intensity curve.
    goal : float, optional
        The desired threshold for encircled energy. The default is 1-1/np.e**2.

    Returns
    -------
    lower_limit: float
        The left coordinate of the bucket which encircles 1-1/e^2 of total
        energy.
    upper_limit: TYPE
        The right coordinate of the bucket which encircles 1-1/e^2 of total
        energy.

    """
    # Find the total area under the curve
    total = simps(intensity, coords)
    
    # Initialize tracked values
    inside = 0
    left_index = np.argmax(intensity)
    right_index = np.argmax(intensity)
    
    # Start looping - narrow left and right index one at a time
    while inside / total < goal:
        if intensity[left_index-1] > intensity[right_index+1]:
            left_index -= 1
        else:
            right_index += 1
        
        # Integrate new bounds and then loop checks if threshold is met
        inside = simps(intensity[left_index:right_index+1],
                       coords[left_index:right_index+1])
        
    lower_limit = coords[left_index]
    upper_limit = coords[right_index]
    
    return lower_limit, upper_limit


def build_steering_bucket_dict():
    """
    Builds the dictionary of bucket sizes for the geometry associated with the
    Steering Order Strehl Ratio test.

    Returns
    -------
    bucket_dict : dictionary of (float, float) tuples, optional
        A dictionary containing the x and y bucket sizes to use at each angle.
        Format is {angle: (size_x, size_y)}, the same as returned in
            build_steering_bucket_dict().
        The default is None.

    """
    bucket_dict = {}
    for angle in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70]:
        bx, by = do_steering_bucket_model(angle)
        bucket_dict[angle] = (bx, by)
    
    return bucket_dict
