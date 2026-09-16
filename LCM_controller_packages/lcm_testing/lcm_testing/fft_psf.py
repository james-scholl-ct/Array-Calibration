"""
A tool for calculating diffraction patterns and estimating wavefront error.

Created on Wed Mar 31 11:03:14 2021.
Modified on Mon Oct 11 2021.

@author: rdpatton
@contributor: jwkuhn

All rights owned by GEOST, Inc.
-----------------------------

References
----------
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    https://johnloomis.org/eop601/notes/matlab/psf/fft_psf.html
    http://kmdouglass.github.io/posts/simple-pupil-function-calculations/
    https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

"""

import csv
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift
from tkinter import Tk
from tkinter.filedialog import askdirectory
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import ndimage
from scipy.special import jv as besselj
from scipy.integrate import simps
from scipy import optimize
from functools import partial
from zernike import RZern

import lcm_testing.ellipse as ellipse


def main(asize, beam_dia, rho=0, wavelength=905e-9, spot_energy=1-1/np.e**2,
         N=1000, zoom=1, astop_px=50, fl=400e-3,
         fcenter=(0, 0), ftheta=0, fphi=0,
         guess=20e-6, wavecos=None, wavefactor=1,
         plot=True, integrate=True, fill_func=None,
         aperture_func='rect',
         aperture_array=[1,    # Number of aperture rows
                         1,    # Number of aperture cols
                         1,    # vspace b/w rows - units of aperture width
                         1,    # hspace b/w cols - units of aperture width
                         ]):
    """
    Calculate diffracted beam spot size.

    A program to calculate spot size for an imaged, diffracted aperture with
    an incident beam described by a generic function. This generic function
    can be de-centered, stretched, expanded and rotated.

    The program also incorporates wavefront error (WFE) modelling by the use
    of Zernike polynomials. Any polynomial with a specified coefficient can
    be specified up to Z14 (UofA indices).

    Parameters
    ----------
    asize : float
        Aperture diameter, in meters.
    beam_dia : float
        Incoming beam width, in meters.
        (Defined as width @ 1/e^2 of peak)
    wavelength : float
        Wavelength of incoming beam, in meters.
        Default is 905 nm.
    spot_energy : float
        Defines the spot size. This is the ratio of total image field energy
        that will be contained within the spot.
        Default is 86.47%.
    N : int, an even number
        Number of computational pixels in each dimension (both aperture plane
        and image field). N should be even; the program adds an extra pixel
        so we have a true center pixel. Simulation creates N+1 sized arrays
        to allow a center pixel.
        Default is 1000.
    zoom : int
        Computational pixel density in the pupil plane.
        To increase image field density for larger-simulated apertures
        (for better gaussian beam modelling), increase pupilzoom.
        WARNING -- Increasing zoom increases computational time exponentially.
        Default is 1.
    astop_px : int
        Number of pixels representing the diameter of the aperture.
        Default is 50.
    fl : float
        Focal length of the lens in the system, in meters.
        Default is 400 mm.
    fcenter : tuple of floats
        Tuple (x, y) describing the centerpoint of the source relative to the
        center of the aperture (optical axis), in meters.
        Default is (0, 0).
    ftheta : float
        Angle by which the incident beam is being stretched.
        Beam diameter stretches (by convention) in the x-axis, prior to
        any rotation, by 1 / cos(ftheta), in degrees.
        Default is 0°.
    fphi : float
        Angle by which the orientation of the incident beam is rotated,
        relative to the x-axis (fphi=0 --> aligned along x-axis), in degrees.
        Default is 0°.
    guess : float
        Guess of the spot size to help speed up integration computation, in
        meters.
        Default is 20 µm.
    wavecos : list of tuples, (int, float)
        List of tuples (UofA index, PV magnitude) with which to build the
        wavefront error.
        Default is None, interpreted as a perfect wavefront.
    wavefactor : float
        Factor by which to multiply all the wavefront errors.
        Default is 1, indicating the Zernike coefficients aren't to be scaled.
    plot : bool
        True or False telling the script whether or not to plot the images
        generated.
        Default is True.
    integrate : bool
        True or False telling the script whether or not to integrate the
        result to find the spot size.
        Default is True.
    fill_func : function
        Function which describes the intensity distribution of the incident
        beam.
        Default is None, which gets translated to fft_psf.gaussian()

    ^^ all distances in meters; all angles in degrees ^^

    """
    # Create wavefront.
    if wavecos is None:
        # Construct a perfect wavefront.
        wavecos = [(0, 0)]
        wavefront = np.zeros((zoom*N+1, zoom*N+1))
    else:
        zern_coeffs = np.zeros(21)
        for ztuple in wavecos:
            z_index, z_cf = zernike_dict(ztuple[0])
            zern_coeffs[z_index] = ztuple[1] * z_cf
        Phi = get_zernike(astop_px, np.asarray(zern_coeffs), wavefactor)
        wavefront = np.zeros((zoom*N+1, zoom*N+1))
        wcen = zoom * N // 2
        x, y = wcen - astop_px // 2, wcen - astop_px // 2
        wavefront[x:x+Phi.shape[0], y:y+Phi.shape[1]] = Phi
        wavefront = np.where(np.isnan(wavefront), 0, wavefront)

    # Create computational grids.
    x_pup, x_ex_rad = create_xs(asize, wavelength, N, astop_px, zoom)

    # Now, compute the PSF and Airy Func with the desired parameters.
    if fill_func is None:
        fill_func = gaussian
    if aperture_func == 'rect':
        aperture = check_rect
    if aperture_func == 'circle':
        aperture = check_circle
    if aperture_func == 'ellipse':
        aperture = check_ellipse
    im_out, ex_out, pup_out = do_calcs(
        x_pup, x_ex_rad,
        asize, rho, beam_dia, wavefront,
        wavelength, N, zoom, astop_px,
        fcenter, ftheta, fphi, fill_func,
        fl, wavecos, wavefactor, aperture,
        aperture_array,
        plot
    )
    print('PSF calculated.')
    
    results = None

    if integrate:
        print(' Performing gaussian curve fit.')
        image = im_out[1]
        xdim_um = im_out[0] * 1e-6

        y_peak_idx, x_peak_idx = ndimage.measurements.center_of_mass(image)
        icen = len(xdim_um) // 2
        cx_um = xdim_um[-1] * (x_peak_idx - icen) / icen * 1e6
        cy_um = xdim_um[-1] * (y_peak_idx - icen) / icen * 1e6
        results = optimize.minimize(
            partial(gaussian_integral,
                    (xdim_um, image),
                    (cx_um*1e-6, cy_um*1e-6)),
            (20e-6, 20e-6, 0),
            method='cobyla',
            constraints=[
                {'type': 'ineq', 'fun': lambda x: x[2]},
                {'type': 'ineq', 'fun': lambda x: 360 - x[2]}],
            options={'rhobeg': 5e-6, 'tol': 1e-9, 'maxiter': 100000},
            bounds=[(0, 100e-6), (0, 100e-6),
                    (0, 360e-6)])

        """
        # Units are currently in µm; change to m for the purpose of getting
        # proper units out in the curve_fit.
        (X, Y) = np.meshgrid(im_out[0] * 1e-6, im_out[0] * 1e-6)
        coords = np.vstack((X.ravel(), Y.ravel()))
        popt, pcov = curve_fit(gaussianxy,
                               coords, im_out[1].ravel(),
                               p0=[guess/4, guess/4, 0, 0, 0],
                               bounds=((3e-6, 3e-6, -40e-6, -40e-6, -180),
                                       (120e-6, 120e-6, 40e-6, 40e-6, 180)))
        print('\nCurve fitting complete.')
        sigma_x = popt[0] * 1e6
        sigma_y = popt[1] * 1e6
        angle = -popt[4]
        xy = (popt[2] * 1e6, -popt[3] * 1e6)
        """
        d_x_um = results['x'][0] * 1e6
        d_y_um = results['x'][1] * 1e6
        xy_um = (cx_um, cy_um)
        angle = results['x'][2] * 1e6
        print(f'Major axis = {d_x_um:.2f} µm, '
              f'Minor axis = {d_y_um:.2f} µm, '
              f'Center at ({xy_um[0]:.3f}, {xy_um[1]:.3f}) µm, '
              # f'Stretched by {popt[3]:.1f}°, '
              f'Rotated by {angle: .2f}°')
        print(f'86.47% Spot Size = {np.pi*d_x_um/2*d_y_um/2:.4f} µm^2')

        fig = plt.gcf()
        axes = fig.get_axes()
        ax = axes[2]
        spot = Ellipse(xy_um, width=d_x_um, height=d_y_um,
                       angle=-angle,
                       color='red', fill=False)
        ax.add_artist(spot)
        
        grid = np.meshgrid(xdim_um, xdim_um)
        mask = ellipse.make_mask(
            grid,
            (xy_um[0]*1e-6, xy_um[1]*1e-6),
            d_x_um/2*1e-6, d_y_um/2*1e-6, angle)
        image = im_out[1]
        masked_image = image * mask
        energy_inside = integration(
            integration(masked_image, xdim_um), xdim_um)
        energy_total = integration(integration(image, xdim_um), xdim_um)
        print(f'Energy contained: {energy_inside / energy_total * 100:.4f} %')

        """
        plt.figure()
        plt.imshow(log_image(masked_image, 3),
                   extent=[im_out[0][0], im_out[0][-1],
                           im_out[0][-1], im_out[0][0]],
                   vmin=0, vmax=1)
        plt.xlim(im_out[0][0], im_out[0][-1])
        plt.ylim(im_out[0][0], im_out[0][-1])
        plt.figure()
        plt.imshow(mask,
                   extent=[im_out[0][0], im_out[0][-1],
                           im_out[0][-1], im_out[0][0]],
                   vmin=0, vmax=1)
        plt.xlim(im_out[0][0], im_out[0][-1])
        plt.ylim(im_out[0][0], im_out[0][-1])
        plt.figure()
        plt.imshow(log_image(image, 3),
                   extent=[im_out[0][0], im_out[0][-1],
                           im_out[0][-1], im_out[0][0]],
                   vmin=0, vmax=1)
        plt.xlim(im_out[0][0], im_out[0][-1])
        plt.ylim(im_out[0][0], im_out[0][-1])
        plt.figure()
        plt.imshow(log_image(image - masked_image, 3),
                   extent=[im_out[0][0], im_out[0][-1],
                           im_out[0][-1], im_out[0][0]],
                   vmin=0, vmax=1)
        plt.xlim(im_out[0][0], im_out[0][-1])
        plt.ylim(im_out[0][0], im_out[0][-1])
        """

    """
    plt.imsave('C:/Users/rdpatton/Documents/Carillon/Beam Spot Size Results'
               '/WFE Plots/testhighres.png',
               image[1],
               cmap='gray')  # metadata={'x-axis': image[0]})
    """
    """
    # Save Figure
    directory = (
        'C:/Users/rdpatton/Documents/Carillon/Beam Spot Size Results/'
        'WFE Plots/Combo RMS/')
    extension = (
        f'Z{ztuple[0]}_gth{ftheta}_gphi{fphi}'
        f'_astop{int(1e3*asize)}mm_gdia{int(1e3*beam_dia)}mm'
        f'_gx{int(1e3*fcenter[0])}mm_gy{int(1e3*fcenter[1])}mm'
        f'_N{N}_apix{astop_px}px_zoom{zoom}x.png'
    )
    filename = directory + extension
    plt.savefig(filename, dpi=900)
    """

    # Save Results  --  currently not using this, due to massive file size for
    #   high res modelling
    # save_results(image, exact, N, astop_px, asize, gsize)

    return results, im_out, ex_out, pup_out


def get_zernike(astop_px, c, wavefactor):
    """
    Build the desired Zernike wavefront.

    Parameters
    ----------
    astop_px : int
        Number of pixels representing the diameter of the aperture.
    c : ndarray
        Array of RMS coefficients with which to build the polynomial(s).
    wavefactor : float
        A global multiplicative factor to scale up or down the overall WFE.

    Returns
    -------
    Phi : 2-D ndarray
        The combined WFE of the superimposed Zernikes.

    """
    cart = RZern(5)
    L, K = astop_px+1, astop_px+1
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    Phi = cart.eval_grid(c, matrix=True)
    Phi = np.where(np.isnan(Phi), np.nan, Phi * wavefactor)
    rms = np.nanstd(Phi - np.nanmean(Phi))
    print(f'Wavefront RMS = {rms:.3e}')

    return Phi


def zernike_dict(ua_index):
    """
    Translate UofA Zernike index to Python zernike index.

    Input a Zernike index (based on UofA indices), and output the index
    that the zernike module uses as well as the RMS-to-Peak-Valley conversion
    factor.

    Parameters
    ----------
    ua_index : int
        The Zernike index according to the UofA convention.

    Returns
    -------
    out : tuple, (int, float)
        Tuple of zernike package index, and the conversion factor that helps
        convert from PV to RMS value when building the Zernike polynomial.

    """
    d = {
        1: (0, 1),                          # Piston
        2: (1, 1 / 4),                      # Tilt
        3: (2, 1 / 4),                      # Tilt
        4: (3, 1 / (2 * np.sqrt(3))),       # Defocus (Power)
        5: (5, 1 / (2 * np.sqrt(6))),       # Astigmatism
        6: (4, 1 / (2 * np.sqrt(6))),       # Astigmatism
        7: (7, 1 / (2 * np.sqrt(8))),       # Coma
        8: (6, 1 / (2 * np.sqrt(8))),       # Coma
        9: (10, 1 / (2 * np.sqrt(5))),      # Spherical
        10: (9, 1 / (2 * np.sqrt(8))),      # Trefoil
        11: (8, 1 / (2 * np.sqrt(8))),      # Trefoil
        12: (11, 1 / (2 * np.sqrt(8))),     # Trefoil
        13: (12, 1 / (2 * np.sqrt(8))),     # Trefoil
        14: (13, 1 / (2 * np.sqrt(8))),     # Trefoil
        15: (14, 1 / (2 * np.sqrt(8))),     # Trefoil
        16: (15, 1 / (2 * np.sqrt(8))),     # Trefoil
        17: (16, 1 / (2 * np.sqrt(8))),     # Trefoil
        18: (17, 1 / (2 * np.sqrt(8))),     # Trefoil
        19: (18, 1 / (2 * np.sqrt(8))),     # Trefoil
        20: (19, 1 / (2 * np.sqrt(8))),     # Trefoil
        21: (20, 1 / (2 * np.sqrt(8))),     # Trefoil
        22: (21, 1 / (2 * np.sqrt(8))),     # Trefoil
        23: (22, 1 / (2 * np.sqrt(8))),     # Trefoil
    }
    out = d[ua_index]
    return out


def zernike_string(wavecos, wavefactor):
    """
    List out the used Zernike polynomial coefficients in a string.

    Parameters
    ----------
    wavecos : list of tuples
        List of tuples representing Zernike index and strength.

    Returns
    -------
    out : str
        String listing out all the Zernike polynomials used in the model.

    """
    out = ''
    for zern in wavecos:
        out += f'Z{zern[0]} = {zern[1]*wavefactor:.3f}, '

    return out[:-2]


def gaussian_integral(im_out, center, x):
    """Wrapper function to allow integration of 2-D Gaussian function
    with x, y coordinates as arrays. Allows the function to be used for
    scipy.optimize.curve_fit fitting.

    Variable x contains the x-width (d_x) and y-width (d_y) of the gaussian
    to be integrated, as well as the phi rotation angle of the gaussian. This
    is the optimizable parameter.

    Center is pre-defined center of the gaussian.
    """
    d_x_um, d_y_um, phi = x
    x0 = center[0]
    y0 = center[1]
    phi = phi * 1e6
    x_dim = im_out[0]
    image = im_out[1]
    spot_energy = 1 - 1/np.e**2
    grid = np.meshgrid(x_dim, x_dim)

    imtotal = integration(integration(image, x_dim), x_dim)

    mask = ellipse.make_mask(grid, (x0, y0), d_x_um/2, d_y_um/2, phi)
    masked_image = mask * image
    energy_inside = integration(integration(masked_image, x_dim), x_dim)

    ratio = energy_inside / imtotal
    error = np.abs(ratio - spot_energy)
    print(f'{error:.4e}\t{d_x_um:.3e}\t{d_y_um:.3e}\t'
          f'{x0:.3e}\t{y0:.3e}\t{phi:.3f}')

    return error


def create_xs(asize, wavelength, N, astop_px, zoom):
    """
    Create the x-axes necessary for model computation.

    Parameters
    ----------
    asize : float
        Aperture diameter, in meters.
    wavelength : float
        Wavelength of incoming beam, in meters.
    N : int
        Number of computational pixels in each dimension.
    astop_px : int
        Number of pixels representing the diameter of the aperture.
    zoom : int
        Computational pixel density in the pupil plane.

    Returns
    -------
    x_pup : 1-D ndarray
        Array representing dimensions in the pupil plane.
        Units in meters.
    x_ex_rad : 1-D ndarray
        Array representing dimensions in the Airy disk plane.
        Units in radians.

    """
    # Displayed image will span 10x the first Airy fringes in all directions.
    x0 = 3.832
    theta0 = np.arcsin(x0 * wavelength / (asize / 2 * 2 * np.pi))
    theta_span = 10 * theta0
    x_ex_rad = np.linspace(-theta_span, theta_span, N+1,
                           endpoint=True)

    pspan = N / astop_px * asize * zoom
    x_pup = np.linspace(-pspan/2, pspan/2, zoom*N+1, endpoint=True)

    return x_pup, x_ex_rad


def do_calcs(x_pup, x_ex_rad,
             asize, rho, beam_dia, wavefront,
             wavelength, N, zoom, astop_px,
             fcenter, ftheta, fphi, fill_function,
             fl, wavecos, wavefactor, aperture,
             aperture_array,
             plot=True):
    """
    Compute the PSF and Airy disk functions.

    Parameters
    ----------
    x_pup : 1-D ndarray

    x_ex_rad : TYPE
        DESCRIPTION.
    asize : TYPE
        DESCRIPTION.
    beam_dia : TYPE
        DESCRIPTION.
    wavefront : TYPE
        DESCRIPTION.
    wavelength : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    zoom : TYPE
        DESCRIPTION.
    astop_px : TYPE
        DESCRIPTION.
    fcenter : TYPE
        DESCRIPTION.
    ftheta : TYPE
        DESCRIPTION.
    fphi : TYPE
        DESCRIPTION.
    fill_function : TYPE
        DESCRIPTION.
    fl : TYPE
        DESCRIPTION.
    zernike_index : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    im_out : TYPE
        DESCRIPTION.
    ex_out : TYPE
        DESCRIPTION.
    pup_out : TYPE
        DESCRIPTION.

    """
    # Build meshgrids
    pgrid = np.meshgrid(x_pup, x_pup)
    exgrid = np.meshgrid(x_ex_rad, x_ex_rad)

    # Construct the pupil function
    if aperture is check_rect:
        m = max(0, 1.3 / np.tan(np.radians(90-abs(rho))) - 0.5)
        w = 16 - 3.072 - m
        foreshortened_w = w * np.sin(np.radians(90-abs(rho)))
        epsilon = 27 / foreshortened_w
    elif aperture is check_ellipse:
        epsilon = np.cos(np.radians(rho)) / np.cos(np.radians(ftheta))
    else:
        epsilon = 1
    
    pupil = make_pupil(pgrid, fill_function, asize, beam_dia, epsilon, aperture,
                       aperture_array,
                       [beam_dia, fcenter[0], fcenter[1], ftheta, fphi, rho])

    # Calculate the PSF.
    image = psf(pupil, wavefront)
    image = image.real  # Image is purely real; just a dtype conversion.

    # Calculate the Airy function based on a perfectly flat wavefront.
    exact = airy_disk(exgrid, asize, wavelength)

    # Rescale x-axes to correspond to physical distances in the image planes.
    x_im_scaled_um, x_ex_scaled_um = rescale_x(x_pup, x_ex_rad,
                                               asize, wavelength, fl,
                                               N, zoom, astop_px)

    # Plot results if desired.
    if plot:
        plot_results(image, exact, pupil, wavefront,
                     x_im_scaled_um, x_ex_scaled_um,
                     beam_dia, fcenter, ftheta, fphi,
                     astop_px, wavecos, wavefactor,
                     aperture_array)

    # Return the results
    im_out = (x_im_scaled_um, image)
    ex_out = (x_ex_scaled_um, exact)
    pup_out = (x_pup, pupil)
    return im_out, ex_out, pup_out


def rescale_x(x_pupil, x_ex_rad, asize, wav, fl, N, zoom, astop_px):
    """
    Rescale the x-axes to correct values of µm based on the model parameters.

    Parameters
    ----------
    x_pupil : 1-D ndarray
        Dimensions used to create the pupil function, in meters.
    x_ex_rad : 1-D ndarray
        Dimensions used to create the exact Airy function, in radians.
    asize : float
        Aperture diameter, in meters.
    wav : float
        Wavelength of the incoming beam, in meters.
    N : int
        Number of computational pixels in each dimension.
    zoom : int
        Computational pixel density in the pupil plane.
    astop_px : int
        Number of pixels representing the diameter of the aperture.
    fl : float
        Focal length of the lens in the system, in meters.

    Returns
    -------
    x_im_scaled_um : 1-D ndarray
        Dimensions that apply to the computed FFT image, in µm.
    x_ex_scaled_um : 1-D ndarray
        Dimensions that apply to the computed Airy disk, in µm.

    """
    # Units of conversion factor are in 1/meters.
    cf = wav * astop_px / (N * zoom * asize * (x_pupil[1] - x_pupil[0]))
    x_im_scaled_um = np.arctan(x_pupil * cf) * fl * 1e6
    x_ex_scaled_um = np.arctan(x_ex_rad) * fl * 1e6

    return x_im_scaled_um, x_ex_scaled_um


def plot_results(image, exact, pupil, wavefront,
                 x_im_scaled, x_ex_scaled,
                 beam_dia, fcenter, ftheta, fphi,
                 astop_px, wavecos, wavefactor,
                 aperture_array):
    """
    Plot the results of the model.

    Note: y is the first index (row) and x is the second index (column)
    in the ndarrays.

    Parameters
    ----------
    image : 2-D ndarray
        FFT of the pupil function, representing the image created by a lens.
    exact : 2-D ndarray
        Exact Airy disk function for the aperture size and wavelength
        specified.
    pupil : 2-D ndarray
        Pupil function we started with.
    wavefront : 2-D ndarray
        Magnitude of wavefront errors at the pupil plane.
    x_im_scaled : 1-D ndarray
        X-values corresponding to the image ndarray, in µm.
    x_ex_scaled : 1-D ndarray
        X-values corresponding to the exact ndarray, in µm.
    beam_dia : float
        Beam diameter of the pupil's fill function.
    fcenter : tuple of floats
        Tuple (x, y) describing the centerpoint of the source
        relative to the center of the aperture (optical axis), in meters.
    ftheta : float
        Angle by which the incident beam is being stretched.
        Beam diameter stretches (by convention) in the x-axis, prior to
        any rotation, by 1 / cos(ftheta), in degrees.
    fphi : float
        Angle by which the orientation of the incident gaussian beam is
        rotated, relative to the x-axis (gphi=0 --> aligned along x-axis),
        in degrees.
    astop_px : int
        Number of pixels representing the diameter of the aperture.
    wavecos : list of tuples
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # ### Sample PSF image ### #
    y_peak_idx, x_peak_idx = ndimage.measurements.center_of_mass(image)
    x_peak_idx = int(round(x_peak_idx, 0))
    y_peak_idx = int(round(y_peak_idx, 0))
    icen = len(image) // 2
    image_samples = len(image) // 4
    image_sampled_x = image[y_peak_idx,
                            x_peak_idx-image_samples:
                            x_peak_idx+image_samples+1]
    x_im_sampled_x = x_im_scaled[x_peak_idx-image_samples:
                                 x_peak_idx+image_samples+1]
    image_sampled_y = image[y_peak_idx-image_samples:
                            y_peak_idx+image_samples+1,
                            x_peak_idx]
    x_im_sampled_y = x_im_scaled[y_peak_idx-image_samples:
                                 y_peak_idx+image_samples+1]
    x_peak = x_im_scaled[-1] * (x_peak_idx - icen) / icen
    y_peak = x_im_scaled[-1] * (y_peak_idx - icen) / icen
    print(f'CoM at ({x_peak:.2f}, {y_peak:.2f}) µm')

    # Sample theoretical image
    zecen = len(exact) // 2
    exact_samples = len(exact) // 4
    x_ex_sampled = x_ex_scaled[zecen-exact_samples:
                               zecen+exact_samples]
    exact_sampled_x = exact[zecen,
                            zecen-exact_samples:
                            zecen+exact_samples]
    exact_sampled_y = exact[zecen-exact_samples:
                            zecen+exact_samples,
                            zecen]

    fig, axes = plt.subplots(2, 3)
    plt.suptitle(
        f'Gcenter at ({fcenter[0]}, {fcenter[1]}) µm, theta={ftheta}°\n'
        f'Zernikes:{zernike_string(wavecos, wavefactor)}'
    )

    # Plot Pupil function
    ax = axes[0][0]
    pcen = len(pupil[0]) // 2
    extent_x = int((astop_px * aperture_array[0] * (1 + aperture_array[2])) / 2)
    extent_y = int((astop_px * aperture_array[1] * (1 + aperture_array[3])) / 2)
    extent = extent_x if extent_x > extent_y else extent_y
    im = ax.imshow(pupil[pcen-extent:pcen+extent,
                         pcen-extent:pcen+extent],
                   extent=[-extent, extent,
                           -extent, extent])
    ax.set_title('Pupil Function')
    fig.colorbar(im, ax=ax)

    # Plot Wavefront
    ax = axes[1][0]
    wcen = len(wavefront[0]) // 2
    im = ax.imshow(wavefront[wcen-extent:wcen+extent,
                             wcen-extent:wcen+extent],
                   extent=[-extent, extent,
                           -extent, extent])
    ax.set_title('Wavefront')
    fig.colorbar(im, ax=ax)

    # Plot x-centerline of both images
    ax = axes[0][1]
    ax.plot(x_ex_sampled, exact_sampled_x,
            linewidth=1,
            label='exact')
    ax.plot(x_im_sampled_x, image_sampled_x,
            linewidth=1,
            label=f'fft: beamwidth={beam_dia * 1e3:.1f} mm')
    ax.set_xlabel('X-position (µm)')
    ax.set_ylabel('Intensity/I_0')
    ax.set_title(f'CoM Plot on X-axis (Y={y_peak:.2f}µm)')
    ax.set_xlim(x_ex_sampled[0], x_ex_sampled[-1])
    ax.legend()

    # Plot y-centerline of both images
    ax = axes[1][1]
    ax.plot(x_ex_sampled, exact_sampled_y,
            linewidth=1,
            label='exact')
    ax.plot(x_im_sampled_y, image_sampled_y,
            linewidth=1,
            label=f'fft: beamwidth={beam_dia * 1e3:.1f} mm')
    ax.set_xlabel('Y position (µm)')
    ax.set_ylabel('Intensity/I_0')
    ax.set_title(f'CoM Plot on Y-axis (X={x_peak:.2f}µm)')
    ax.set_xlim(x_ex_sampled[0], x_ex_sampled[-1])
    ax.legend()

    # Use logrithmic transformation for output images
    # Plot PSF from pupil function
    ax = axes[0][2]
    ax.imshow(image,
        #log_image(image, 3),
              extent=[x_im_scaled[0], x_im_scaled[-1],
                      x_im_scaled[-1], x_im_scaled[0]],
              cmap='jet')
    ax.set_xlabel('X-position (µm)')
    ax.set_ylabel('Y-position (µm)')
    ax.set_title('PSF based on Pupil')
    ax.set_xlim(x_ex_scaled[0], x_ex_scaled[-1])
    ax.set_ylim(x_ex_scaled[0], x_ex_scaled[-1])
    ax.scatter(
        x_peak, y_peak,
        color='red', marker='+', linewidth=1, alpha=0.75,
        label='Center of Mass'
    )
    ax.legend()

    # Exact Bessel function
    ax = axes[1][2]
    ax.imshow(exact,
        #log_image(exact, 3),
              extent=[x_ex_scaled[0], x_ex_scaled[-1],
                      x_ex_scaled[0], x_ex_scaled[-1]],
              cmap='jet')
    ax.set_xlabel('X-position (µm)')
    ax.set_ylabel('Y-position (µm)')
    ax.set_title('Exact Diffraction w/Tophat Fill & Zero WFE')
    ax.set_xlim(x_ex_scaled[0], x_ex_scaled[-1])
    ax.set_ylim(x_ex_scaled[0], x_ex_scaled[-1])

    fig.set_size_inches(15, 13)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


def save_results(image, exact, asize, beam_dia, N, astop_px):
    """
    Save FFT image to CSV file.

    Turns out not to be very useful with large arrays due to array size.
    N=2000 with pupilzoom=3 gives >750MB file.

    Parameters
    ----------
    image : 2-D ndarray
        The fft of the pupil function, representing the image created by a
        lens.
    exact : 2-D ndarray
        The exact Airy disk function for the aperture size and wavelength
        specified.
    asize : float
        The aperture diameter, in meters.
    beam_dia : float
        Incoming beam width, in meters.
    N : int
        Number of computational pixels in each dimension.
    astop_px : int
        Number of pixels representing the diameter of the aperture.

    Returns
    -------
    None.

    """
    root = Tk()
    directory = askdirectory()
    root.destroy()
    filename = (
        directory +
        f'/N{N}_{astop_px}px_{int(1e3*asize)}mm_'
        '{int(1e3*beam_dia)}mm.csv'
    )
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow('PSF X-Coords')
        writer.writerow(image[0])
        writer.writerow('')
        writer.writerow('PSF Amplitudes')
        writer.writerows(image[1])


def compute_units(exact, wav, asize):
    """
    Compute unit conversion factor in the image field.

    Prints out the point along the exact x-axis at which the first minimum
    occurs.

    Uses based on bessel zero, wavelength and ap stop diameter.

    Parameters
    ----------
    exact : 2-D ndarray
        The computed Airy function.
    wav : float
        The wavelength of the beam, in meters.
    asize : float
        The diameter of the aperture, in meters.

    Returns
    -------
    cf : float
        A conversion factor between x-index and radians, in units of radians
        per index value
    """
    ymid = len(exact) // 2
    zcen_exact = exact[:, ymid]
    x0 = 3.832  # numerical solution for first zero of jinc function

    # Find the first minimum.
    zhalf_idx = len(zcen_exact) // 2
    zhalf = zcen_exact[zhalf_idx:]
    i = 1
    minval = zhalf[0]
    while minval > zhalf[i]:
        minval = zhalf[i]
        i += 1

    # Calculate first Airy Disk minimum, in radians.
    theta0 = np.arcsin(x0 * wav / (asize * 2 * np.pi))
    minidx = i - 1

    # i is index of theta0
    cf = theta0 / minidx
    print(f'Check: minval = {minval} at {minidx} pixels from center.')
    print(f'Spot size = {theta0*2} radians')
    print(f'radians per pixel in exact image frame: {cf}')

    return cf


def integration(z, x=None):
    """
    Integrate a function.

    Uses Simpson integration of a function z, over coordinates x.

    Parameters
    ----------
    z : array_like
        The function values to integrate. In general, this can be an ndarray,
        such that a 2-D array can be passed, in which case this function
        returns a 1-D array of values along one axis.
    x : 1-D ndarray, optional
        The x-coordinates corresponding to values in z. The default is None.

    Returns
    -------
    out : array_like
        The result of integration. The shape of this value depends on the
        shape of the input array, and will be one dimension smaller than the
        input array.
    """
    out = simps(z, x)
    return out


def tophat(grid, *args):
    """
    Return a 1 for all points, so a tophat function can be used.

    Parameters
    ----------
    grid : numpy meshgrid
        The XY cordinates on which the tophat function is built.
    *args : N/A
        Argument exists in the function definition for interfacing with the
        make_pupil function used to call this. Other fill functions use
        parameters to calculate values, whereas this one equals 1 everywhere.

    Returns
    -------
    out : ndarray
        An ndarray with the same dimensions as the incoming meshgrid, where
        all elements equal 1.

    """
    out = np.where(grid is grid, 1, 1)
    return out


def slant(grid, deviation, *args):
    """
    Returns a slanted pupil intensity, currently only in the x-axis, where
    `deviation` represents the amount of % shift per m in the astop.
    E.g., 10µm astop width and 10e6 deviation gives 1% deviation across astop.

    Parameters
    ----------
    grid : numpy meshgrid
        The XY cordinates on which the tophat function is built.
    *args : N/A
        Argument exists in the function definition for interfacing with the
        make_pupil function used to call this. Other fill functions use
        parameters to calculate values, whereas this one equals 1 everywhere.

    Returns
    -------
    out : ndarray
        An ndarray with the same dimensions as the incoming meshgrid, where
        all elements equal 1.

    """
    X, Y = grid
    # Deviation is % per meter, this way this function can be invariant to 
    # aperture size.
    peak = deviation
    
    B = (1 - peak * X)
    
    # Zero out any negative values.
    B = np.where(B < 0, 0, B)
    
    return B


def gaussian(grid, beam_dia, cx, cy, theta, phi, rho):
    """
    Generate a gaussian beam grid based on input parameters.

    Parameters
    ----------
    grid : numpy meshgrid
        The XY coordinates on which the gaussian is built, in meters.
    *args: list of arguments
        The beam_dia, Center, Theta and Phi which characterize the Gaussian.

    Arguments
    ---------
        *beam_dia : float
            The beam_dia of the beam, in m.
        *cx, cy : pair of floats
            The xy coordinates of the center of the beam, in m, represented as
            two variables
        *theta : float
            The forelengthening angle of the beam in the direction of the major
            axis, in degrees. I.e., the angle at which the beam is incident
            upon the aperture.
            E.g., 60° widens the beam by a factor of 2 along the major axis.
        *phi : float
            The angle of rotation of the beam's major axis, in degrees, CCW
            relative to the positive x-axis.
            E.g., 0° aligns the major axis with the positive x-axis on the
            grid.  90° aligns the major axis with the positive y-axis.
        *rho : float
            The exit angle of the beam - accounts for foreshortening of the
            beam width in the x-direction. A stretched beam due to high
            incident angle will be compressed by a factor of
            1 / cos(rho). E.g. if incident angle is -70° and exit angle is
            +70°, then the outgoing beam sigma_x is the same as the incoming
            beam sigma_x.

    Returns
    -------
    B : 2-D ndarray
        Array of floats describing the intensity of the Gaussian beam
        on the XY coordinate grid.
    """
    X, Y = grid
    x0, y0 = cx, cy
    g_scale = beam_dia / 4  # gaussian characteristic sigma (diameter = 4 sigma)
    sig_y = g_scale
    sig_x = g_scale * np.cos(np.radians(rho)) / np.cos(np.radians(theta))
    phi = np.radians(phi)
    if (sig_x - sig_y) < 1e-12:
        phi = 0

    a = np.cos(phi)**2 / (2 * sig_x**2) + np.sin(phi)**2 / (2 * sig_y**2)
    b = -np.sin(2 * phi) / (4 * sig_x**2) + np.sin(2 * phi) / (4 * sig_y**2)
    c = np.sin(phi)**2 / (2 * sig_x**2) + np.cos(phi)**2 / (2 * sig_y**2)

    B = (1 / (2 * np.pi * sig_x * sig_y)
         * np.exp(-(a * (X-x0)**2 + 2 * b * (X-x0) * (Y+y0) + c * (Y+y0)**2)))
    B = B / np.max(B)

    return B


def gaussianxy(grid, sig_x, sig_y, cx, cy, phi):
    """
    Generate a gaussian beam grid based on input parameters.

    Parameters
    ----------
    grid : numpy meshgrid
        The XY coordinates on which the gaussian is built, in meters.
    *args: list of arguments
        The beam diameter, Center, Theta and Phi which characterize the Gaussian.

    Arguments
    ---------
        *width : float
            The width of the beam, defined as 4σ wide.
        *center : tuple of floats
            The xy coordinates of the center of the beam, in m, represented as
            a tuple, (x, y).
        *theta : float
            The forelengthening angle of the beam in the direction of the
            major axis, in degrees.
            E.g., 60° widens the beam by a factor of 2 along the major axis.
        *phi : float
            The angle of rotation of the beam's major axis, in degrees, CCW
            relative to the positive x-axis.
            E.g., 0° aligns the major axis with the positive x-axis on the
            grid.  90° aligns the major axis with the positive y-axis.

    Returns
    -------
    B : 2-D ndarray
        Array of floats describing the intensity of the Gaussian beam
        on the XY coordinate grid.
    """
    X, Y = grid
    x0, y0 = cx, cy
    phi = np.radians(phi)
    if (sig_x - sig_y) < 1e-12:
        phi = 0

    a = np.cos(phi)**2 / (2 * sig_x**2) + np.sin(phi)**2 / (2 * sig_y**2)
    b = -np.sin(2 * phi) / (4 * sig_x**2) + np.sin(2 * phi) / (4 * sig_y**2)
    c = np.sin(phi)**2 / (2 * sig_x**2) + np.cos(phi)**2 / (2 * sig_y**2)

    B = (1 / (2 * np.pi * sig_x * sig_y)
         * np.exp(-(a * (X-x0)**2 + 2 * b * (X-x0) * (Y+y0) + c * (Y+y0)**2)))
    B = B / np.max(B)

    return B.ravel()


def make_pupil(meshgrid, fill_func, asize, beam_dia, epsilon, aperture,
               aperture_array,
               args):
    """
    Generate the pupil function with the specified shape and a specified
    filling function.

    Parameters
    ----------
    pupilgrid : numpy meshgrid
        XY coordinates of coordinates on which to build the pupil function.
    fill_func : function
        A function that accepts a meshgrid and a set of arguments to build
        a beam profile describing the intensity of the beam incident on the
        aperture.
    asize : float
        Diameter of the aperture, in meters.
    aperture : func
        The function describing how to build the shape of the aperture.
    args : list of arguments
        A list of arguments characterizing the fill_func.

    Returns
    -------
    pupil : 2-D ndarray
        The pupil function (aperture mask * incoming beam).
    """
    # Unpack aperture array details
    aperture_rows = aperture_array[0]
    aperture_cols = aperture_array[1]
    aperture_vspace = aperture_array[2]
    aperture_hspace = aperture_array[3]
    
    # Calculate aperture 
    y0 = -(aperture_rows - 1) / 2 * asize * (1 + aperture_vspace)
    x0 = -(aperture_cols - 1) / 2 * asize * (1 + aperture_hspace)
    dy = asize * (1 + aperture_vspace)
    dx = asize * (1 + aperture_hspace)
    
    # Set up base array
    ap_mask = np.zeros(meshgrid[0].shape, dtype='int')
    x = x0
    y = y0
    
    for row in range(aperture_rows):
        for col in range(aperture_cols):
            temp_mask = np.where(
                aperture(meshgrid, asize, epsilon, center=(x, y)),
                1, 0)
            #ap_mask = np.where(ap_mask + temp_mask > 0, 1, 0)
            ap_mask = ap_mask + temp_mask
            x += dx
        x = x0
        y += dy

    B = fill_func(meshgrid, *args)
    pupil = ap_mask * B
    
    return pupil


def check_rect(point, asize, epsilon, center=(0, 0)):
    # Checks if the point (x, y) is inside a rectangle described by the other
    # parameters.
    x, y = point

    return (
        (np.abs(x - center[0]) <= asize / 2 / epsilon) *
        (np.abs(y - center[1]) <= asize / 2) == 1
    )


def check_circle(point, asize, epsilon, center=(0, 0)):
    x, y = point
    
    return np.sqrt((x-center[0])**2 + (y-center[1])**2) <= asize / 2


def check_ellipse(point, asize, epsilon, center=(0, 0)):
    cx, cy = center
    ry = asize / 2
    rx = asize / 2 / epsilon
    return ellipse.check_inside(point, cx, cy, rx, ry, 0)


def rect(grid, width, height, cx, cy, phi):
    """
    Generate a grid fill in the shape of a rectangle.
    CURRENTLY UNUSED.
    
    Rectangle is centered at (cx, cy), with width and height dimensions, and
    rotated about the center point by phi.
    
    >> Currently ignoring phi <<
    """
    X, Y = grid
    Bx = np.where(X < cx + width / 2 and X > cx - width / 2, 1, 0)
    By = np.where(Y < cy + height / 2 and Y > cy - height / 2, 1, 0)
    B = np.where(Bx and By, 1, 0)
    return B


def psf(pupil, w):
    """
    Calculate the PSF using Fraunhofer diffracion.

    In other words, the PSF is the Fourier Transform of the pupil's wavefront.

    It is necessary to shift, then FFT, then shift back. The final result is
    normalized relative to the peak of the PSF. Eventually, it may make sense
    to pass in a total power parameter to this function, and normalize to a
    specific unit of power (e.g. mW).

    Since the system is using a lens to focus the diffracted beam, we can
    consider this as a far-field interaction, which allows us to use the
    Fraunhofer approximation, which is equivalent to taking the FFT of the
    pupil function.

    Parameters
    ----------
    pupil : 2-D ndarray
        The pupil function to be FFT'd.
    w : 2-D ndarray
        The wavefront error of the beam incident on the aperture.

    Returns
    -------
    im : 2-D ndarray
        The image formed in the far-field.

    """
    wfpupil = pupil * np.exp(2 * np.pi * 1j * w)
    psf_a = fftshift(fft2(ifftshift(wfpupil)))
    im = np.abs(psf_a)**2
    im = im / np.max(im)
    return im


def log_image(image, decades):
    """
    Transform the input to a logarithmically-scaled output.

    Values that would be negative using the parameters are forced to zero.

    Parameters
    ----------
    image : 2-D ndarray
        The image to be re-scaled, values ranging from 0 to 1.
    decades : int
        The number of decades on which to scale the output. A larger value
        has the effect of making smaller features visible on the final plots.

    Returns
    -------
    out : 2-D ndarray
        The re-scaled image.
    """
    out = 1 + np.log10(image) / decades
    out = np.where(out > 0, out, 0)
    return out


def airy_disk(grid, asize, wavelength):
    """
    Compute a 2-D Airy disk from the input grid, aperture size and wavelength.

    Parameters
    ----------
    grid : numpy meshgrid
        The XY coordinates on which the Airy disk is built.
    asize : float
        The size of the aperture to diffract, in meters.
    wavelength : float
        The wavelength of light incident on the aperture, in meters.

    Returns
    -------
    result : 2-D ndarray
        The computed 2-D Airy function based on the supplied parameters.

    """
    X, Y = grid
    R = np.sqrt(X**2 + Y**2)
    Z = 2 * np.pi * R * asize / 2 / wavelength
    result = np.where(R > 0, (2 * besselj(1, Z) / Z) ** 2, 1)
    return result
