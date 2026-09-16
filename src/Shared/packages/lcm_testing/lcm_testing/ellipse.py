# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:22:15 2021

@author: rdpatton

References: https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio
"""

import numpy as np



def ellipse(phis, center, rx, ry, theta):
    """
    A function to return parameterized values for a rotated, offcenter ellipse

    Note: the major radius of the unrotated ellipse lies along the x-axis.

    Parameters
    ----------
    phis : the parameter, a set of angles to map to an ellipse, 0 to 2pi
           Should be a numpy array.
    center : the centerpoint of the ellipse, (cx, cy)
    rx : the major radius
    ry : the minor radius
    theta : the ellipse rotation angle, measured from the x-axis, in degrees

    Returns
    -------
    A numpy ndarray [X, Y] of coordinates defining the ellipse.

    """
    theta = np.radians(theta)
    cx, cy = center
    X = rx * np.cos(phis) * np.cos(theta) - ry * \
        np.sin(phis) * np.sin(theta) + cx
    Y = rx * np.cos(phis) * np.sin(theta) + ry * \
        np.sin(phis) * np.cos(theta) + cy

    # return np.asarray([(x, y) for x, y in zip(X, Y)])
    return np.asarray([X, Y])


def check_inside(point, cx, cy, rx, ry, theta):
    # Checks if the point (x, y) is inside the ellipse described by the other
    # parameters.
    x, y = point
    theta = np.radians(theta)
    
    dist = (
        ((x - cx) * np.cos(theta) - (y + cy) * np.sin(theta))**2 / rx**2
        + ((x - cx) * np.sin(theta) + (y + cy) * np.cos(theta))**2 / ry**2)

    return dist <= 1


def make_mask(grid, center, rx, ry, theta):
    """
    Returns a mask where points enclosed or intersected by the ellipse are 1
    and points outside the ellipse are 0.

    Parameters
    ----------
    cf : The conversion factor between elements and ellipse parameters, in
         pixels per ellipse unit.
    shape : The shape (size) of the mask, in the units of the ellipse
            parameters.
    center : The centerpoint of the ellipse, (cx, cy)
    rx : The major radius
    ry : The minor radius
    theta : The ellipse rotation angle, measured from the x-axis, in degrees

    Returns
    -------
    A numpy ndarray of <shape> shape, with 1s inside the ellipse and 0
    otherwise.

    """
    cx, cy = center
    mask = np.where(check_inside(grid, cx, cy, rx, ry, theta), 1, 0)

    return mask
