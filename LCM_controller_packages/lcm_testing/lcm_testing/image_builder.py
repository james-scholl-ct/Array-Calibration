# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:19:37 2021

@author: rdpatton
"""

import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import Akima1DInterpolator
from scipy.integrate import simps

import lcm_testing.fft_psf as fft_psf


PLOT_DICT = {-60: 0,
             -50: 1,
             -40: 2,
             -30: 3,
             -20: 4,
             -10: 5,
             0: 6,
             10: 7,
             20: 8,
             30: 9,
             40: 10,
             50: 11,
             60: 12,
             70: 13}


POS_DICT = {'a': 0,
            'b': 2.5,
            'c': 5,
            'd': 7.5,
            'e': 10,
            'f': 12.5,
            'g': 15,
            'h': 17.5,
            'i': 20,
            'j': 22.5,
            'k': 25}

SN_DICT = {'mirror': 'red',
           '1567': 'blue',
           '1577': 'green',
           '1601': 'orange',
           '1603': 'brown'}


def plot_specular_scans(base_path):
    fwhms = []
    stdevs = []
    positions = []
    sns = []
    path = base_path + '*.txt'
    
    for f in glob.glob(path):
        results = get_fwhm(f)
        fwhms.append(results[0])
        stdevs.append(results[1])
        positions.append(POS_DICT[(f[:-4].split('_')[1])])
        if 'mirror' in f:
            sns.append('mirror')
        else:
            sns.append(f[-10:-6])
    
    # Organize data into DataFrame
    data = pd.DataFrame(data=[positions, sns,
                              [fwhm[0] for fwhm in fwhms],
                              [fwhm[1] for fwhm in fwhms],
                              [stdev[0] for stdev in stdevs],
                              [stdev[1] for stdev in stdevs]]).transpose()
    data.columns = ['Position', 'SN',
                    'FWHM x', 'FWHM y',
                    'Stdev x', 'Stdev y']
    data.sort_values(['SN', 'Position'], inplace=True)
    
    # Plot each SN data one series at a time
    z_diffs = []
    fig, axes = plt.subplots(2)
    unique_sns = list(set(sns))
    unique_sns.sort()
    for sn in unique_sns:
        if sn == 'mirror':
            label = 'Mirror'
        else:
            label = f'SN00{sn}'
        series = data.loc[data['SN'] == sn]
        axes[0].plot(series['Position'], series['FWHM x'],
                     label=f'{label}', linewidth=1,
                     color=SN_DICT[sn], linestyle='solid')
        axes[1].plot(series['Position'], series['FWHM y'],
                     label=f'{label}', linewidth=1,
                     color=SN_DICT[sn], linestyle='solid')
    
        # Use Akima spline to interpolate into a smoother curve, to find
        # minimum (best saggital / tangential focus)
        x = np.linspace(0, 25, 1000)
        interpolator = Akima1DInterpolator(series['Position'],
                                           series['FWHM x'])
        interpolated_x = interpolator(x)
        axes[0].plot(x, interpolated_x,
                     color=SN_DICT[sn], linestyle='dotted')
        interpolator = Akima1DInterpolator(series['Position'],
                                           series['FWHM y'])
        interpolated_y = interpolator(x)
        axes[1].plot(x, interpolated_y,
                     color=SN_DICT[sn], linestyle='dotted')
        
        astig_z_diff = abs(x[np.argmin(interpolated_x)] - x[np.argmin(interpolated_y)])
        z_diffs.append(astig_z_diff)
    
    # Plot sagittal / tangential focii difference
    plt.figure()
    plt.bar(unique_sns, z_diffs)
    plt.title('Z-diff in Sagittal and Tangential minima')
    plt.xlabel('z-distance (mm)')
        
    
    # Plot the theroetical values
    fwhms_x, fwhms_y, buckets_x, buckets_y = get_defocus_fwhm(10e-3)
    axes[0].plot([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25], fwhms_x,
                 label='Theoretical Perfect w/Defocus',
                 color='black', linestyle='solid')
    axes[1].plot([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25], fwhms_y,
                 label='Theoretical Perfect w/Defocus',
                 color='black', linestyle='solid')
            
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title('X-width')
    axes[1].set_title('Y-width')
    fig.suptitle('Specular Reflection (10° from normal)'
                 'w/z-axis translation through focus')
    axes[0].set_xlabel('z-distance on stage (mm)')
    axes[1].set_xlabel('z-distance on stage (mm)')
    axes[0].set_ylabel('FWHM in x-profile (µm)')
    axes[1].set_ylabel('FWHM in y-profile (µm)')


def get_defocus_fwhm(asize):
    fwhms_x = []
    fwhms_y = []
    buckets_x = []
    buckets_y = []

    for waves in np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) * 0.2158:
        _, result, _, _ = fft_psf.main(asize, 64e-3, astop_px=200, N=2000, zoom=1,
                                       integrate=False, plot=False,
                                       wavecos=[(4, waves)],
                                       aperture_func='circle')
        coords = result[0]
        image = result[1]
        
        # Find peak
        idx = image.argmax() % image.shape[1]
        idy = image.argmax() // image.shape[1]
        
        horiz_slice = image[idy]
        vert_slice = [im[idx] for im in image]
        
        fwhm_x = find_fwhm(coords, horiz_slice)
        fwhm_y = find_fwhm(coords, vert_slice)
        
        fwhms_x.append(fwhm_x)
        fwhms_y.append(fwhm_y)
        
        buckets_x.append(integrate_bucket(coords, horiz_slice))
        buckets_y.append(integrate_bucket(coords, vert_slice))
        print(fwhm_x, fwhm_y)
    
    return fwhms_x, fwhms_y, buckets_x, buckets_y


def integrate_bucket(coords, intensity, left, right):
    coordslice = coords[left:right+1]
    intensityslice = intensity[left:right+1]
    
    return simps(intensityslice, coordslice)
    

def find_fwhm(coords, intensity):
    peak = np.max(intensity)  # Find the maximum y value
    
    # Find left index
    left_index = 0
    while intensity[left_index] < peak / 2:
        left_index += 1
    
    # Find right index
    right_index = len(coords) - 1
    while intensity[right_index] < peak / 2:
        right_index -= 1
    
    # Left interpolation
    x1 = coords[left_index+1]
    x2 = coords[left_index]
    y1 = intensity[left_index+1]
    y2 = intensity[left_index]
    m = (y2 - y1) / (x2 - x1)
    X_left = (peak / 2.0 - y1) / m + x1
    
    # Right interpolation
    x1 = coords[right_index-1]
    x2 = coords[right_index]
    y1 = intensity[right_index-1]
    y2 = intensity[right_index]
    m = (y2 - y1) / (x2 - x1)
    X_right = (peak / 2.0 - y1) / m + x1
    
    
    
    fwhm = X_right - X_left
    return fwhm


def get_steered_fwhm(path):
    fwhms = []
    stdevs = []
    angles = []
    sns = []
    
    for f in glob.glob(path):
        results = get_fwhm(f)
        fwhms.append(results[0])
        stdevs.append(results[1])
        angles.append(float(f[:-4].split('_')[1]))
        if 'mirror' in f:
            sns.append('mirror')
        else:
            sns.append(f[-10:-6])
        
    return angles, sns, fwhms, stdevs


def plot_steered_fwhms(sn, base_path):
    path = base_path + 'sn' + sn + '*'
    angles, sns, fwhms, stdevs = get_steered_fwhm(path)
    
    fig, axes = plt.subplots(2)
    for angle, sn, fwhm, stdev in zip(angles, sns, fwhms, stdevs):
        axes[0].scatter(angle, fwhm[0],
                        s=2, label=f'SN00{sn}, x-FWHM')
        axes[1].scatter(angle, fwhm[1],
                        s=2, label=f'SN00{sn}, y-FWHM')
    #plt.legend()


def get_fwhm(filename):
    pdata = pd.read_csv(filename,
                        skiprows=lambda x: x not in np.arange(35, 51),
                        encoding='latin-1')
    
    fwhm_x = float(pdata['Mean'].iloc[4])
    fwhm_y = float(pdata['Mean'].iloc[5])
    fwhm_x_stdev = float(pdata['S. Dev.'].iloc[4])
    fwhm_y_stdev = float(pdata['S. Dev.'].iloc[5])
    
    return [(fwhm_x, fwhm_y), (fwhm_x_stdev, fwhm_y_stdev)]


def execute(filename, indiplot=False, normalize=False):
    pdata = pd.read_csv(filename, usecols=[0, 1, 2, 3], skiprows=55,
                        encoding='latin-1')
    
    X = np.asarray(pdata[r'Position X ROI#1'].dropna())
    Y = np.asarray(pdata[r'Position Y ROI#1'].dropna())
    xi = np.asarray(pdata[r' Intensity X ROI#1'].dropna())
    yi = np.asarray(pdata[r' Intensity Y ROI#1'].dropna())
    
    image = build_image(xi, yi)
    
    if normalize:
        image = image / np.max(image)
    
    if indiplot:
        plt.figure()
        plt.imshow(image,
                   extent=[X[0], X[-1],
                           Y[0], Y[-1]],
                   cmap=plt.get_cmap('jet'))
        plt.xticks([X[0], X[-1]])
        plt.yticks([Y[0], Y[-1]])
        try:
            f = filename[:-4].split('\\')[-1]
            fa = f.split('_')
            angle = fa[1]
            order = fa[2]
            plt.title(f'{angle}° angle; Steering Order {order}')
        except:
            pass
    
    return (X, Y, image)


def grab_all_spots(sn, base_path):
    images = []
    angles = []
    orders = []
    
    path = base_path + f'SN00{sn}*'

    for f in glob.glob(path):
        angles.append(int(f.split('_')[1]))
        images.append(execute(f))
        orders.append(int(f.split('_')[2].split('.')[0]))

    fig, axes = plt.subplots(14)
    
    impeak = 0
    for image in images:
        if np.max(image[2]) > impeak:
            impeak = np.max(image[2])
    
    for data, angle, order in zip(images, angles, orders):
        dest = PLOT_DICT[angle]
        ax = axes[dest]
        X = data[0]
        Y = data[1]        
        image = data[2]
        ax.imshow(image, extent=[X[0], X[-1],
                                 Y[0], Y[-1]],
                  cmap=plt.get_cmap('jet'),
                  label=f'{angle}°',
                  vmin=0, vmax=impeak)
        ax.set_xlim([0, 3500])
        if sn == 1567 and angle > -40:
            ax.set_ylim([Y[0], Y[0]+150])
        elif sn == 1567 and angle == -40:
            ax.set_ylim([Y[0]+25, Y[0]+175])
        elif sn == 1567:
            ax.set_ylim([Y[0]+50, Y[0]+200])
        else:
            ax.set_ylim([Y[0], Y[0]+150])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_xticks([X[0], X[len(X)//2], X[-1]])
        #ax.set_yticks([Y[0], Y[len(Y)//2], Y[-1]])
        ax.set_ylabel(f'{angle}°')#' angle; Steering Order {order}')
        for item in ([ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    ax.set_xticks([0, 3500])
    ax.xaxis.label.set_fontsize(20)
    fig.suptitle(f'SN00{sn} Spot Profiles (all x, y values in µm)'
                 '\nHeight: 150µm in all plots')
    fig._suptitle.set_fontsize(30)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    


def z_scan_gif(base_path):
    

    images = []
    positions = []
    
    # Do LCM profiles
    path = base_path + '\\SN001601*'

    for f in glob.glob(path):
        images.append(execute(f))
        positions.append(POS_DICT[(f[:-4].split('_')[1])])
        #sn = f[f.index('SN')+2:f.index('SN')+8]
    
    temp = images.copy()
    temp.reverse()
    images = images + temp
    temp = positions.copy()
    temp.reverse()
    positions = positions + temp

    fig = plt.figure()
    ax = plt.axes(xlim=(images[0][0][0], images[0][0][0]+400),
                  ylim=(images[0][1][0], images[0][1][0]+400))
    im = plt.imshow(images[0][2], cmap='jet',
                    extent=[images[0][0][0], images[0][0][0]+400,
                            images[0][1][0], images[0][1][0]+400],
                    label=f'Mirror at {positions[0]} mm')
    #ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
    #ax.set_title(f'SN {sn} at {positions[0]} mm')

    def init():
        im.set_data(images[0][2])
        return [im]
    
    def animate(i):
        im.set_array(images[i][2])
        #ttl.set_text(f'SN {sn} at {positions[i]} mm')
        plt.show()
        return [im]

    ani = FuncAnimation(fig, animate, frames=len(images),
                        init_func=init, blit=True, interval=250)
    plt.show()
    ani.save('zscan.gif')
    
    
    """
    for data, position in zip(images, positions):
        dest = (POS_DICT[position], 0)
        ax = axes[dest[0], dest[1]]
        X = data[0]
        Y = data[1]
        image = data[2]
        ax.imshow(image, extent=[X[0], X[-1],
                                 Y[0], Y[-1]],
                  cmap=plt.get_cmap('jet'),
                  label=f'{position} mm')
        ax.set_xlim([X[0], X[0]+400])
        ax.set_ylim([Y[0], Y[0]+400])
        ax.set_xticks([X[0], X[-1]])
        ax.set_yticks([Y[0], Y[-1]])
        ax.set_title(f'SN{sn} @ {position} mm')
    
    # Do Mirror profiles
    path = folder_path + '\\mirror*'

    for f in glob.glob(path):
        images.append(execute(f))
        positions.append(int(f[:-4].split('_')[1]))

    for data, position in zip(images, positions):
        dest = (POS_DICT[position], 1)
        ax = axes[dest[0], dest[1]]
        X = data[0]
        Y = data[1]
        image = data[2]
        ax.imshow(image, extent=[X[0], X[-1],
                                 Y[0], Y[-1]],
                  cmap=plt.get_cmap('jet'),
                  label=f'{position} mm')
        ax.set_xlim([X[0], X[0]+400])
        ax.set_ylim([Y[0], Y[0]+400])
        ax.set_xticks([X[0], X[-1]])
        ax.set_yticks([Y[0], Y[-1]])
        ax.set_title(f'{position} mm')
    
    fig.suptitle(f'SN{sn} Scan Through Focus vs Mirror (all x, y values in µm)')
    """


def build_image(xi, yi):
    yivert = np.vstack(yi)
    
    image = xi * yivert
    
    return image
