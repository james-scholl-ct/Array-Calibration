# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:47:29 2021

@author: rdpatton

Source(s):
    https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
    https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    https://setosa.io/ev/image-kernels/
    https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import imutils
import cv2
import glob
import time
import random


def toGray(a):
    if len(a.shape) > 1:
        return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return None


def shiftAndBlend(a, b, prevYShift, plotIndex, totalPlots,
                  plotStats=True):
    #(a, b) = images
    
    # Convert stuff to grayscale
    if len(a.shape) > 2:
        agray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    else:
        agray = a
    if len(b.shape) > 2:
        bgray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    else:
        bgray = b
        
    # Convert images to floats to allow smooth blending
    A = agray#.astype('float32')
    B = bgray#.astype('float32')
    shiftdiff = B.shape[0] - A.shape[0]
    if shiftdiff != 0:
        if prevYShift > 0:
            B = B[:-shiftdiff, :]
        else:
            B = B[shiftdiff:, :]
    
    # Define sliding bounds
    yslidemin = -20
    yslidemax = 20
    xslidemin = B.shape[1] // 3
    xslidemax = B.shape[1]
    
    # Create storage array for computed |x^2| values at each shift
    x2 = np.zeros((yslidemax - yslidemin + 1, xslidemax - xslidemin + 1),
                  dtype='float32')
    
    # Calculate the total number of calculations for the progess bar
    numcalcs = (xslidemax - xslidemin + 1) * (yslidemax - yslidemin + 1)
    count = 0
    starttime = time.time()
    prevtime = time.time()
    
    print('\nBeginning Overlap Computations')
    # Slide the second image over the first, pixel by pixel, and compute
    # the difference.
    # We necessarily need the width of image B to be less than or equal to
    # the width of image A.
    # The counter "i" is the width of the overlap, in pixels.
    # Assume the images overlap by at least a quarter of image B's width
    for i in np.arange(xslidemin, xslidemax + 1):
        # Also slide the images vertically, thereby excluding "j" pixels
        # from the top of one image and the bottom of the other,
        # or vice versa.
        # Negative j moves the B image up, relative to the A image.
        for j in np.arange(yslidemin, yslidemax + 1):
            # First, create a copy of the image
            X = A.copy()
            Y = B.copy()
            
            # Trim the top/bottom rows from the two images according to
            # the "j" shift amount.
            X, Y = trimRows(X, Y, j)
            
            # Now, subtract the left slice of image B, of width i, from the
            # right slice of image A, of width i.
            X[:, -i:] -= Y[:, :i]
                    
            # Remove the left part of the image, which has not been overlapped
            X = X[:, -i:]
            
            # Compute the average of the would-be overlapped image
            Atemp, Btemp = trimRows(A, B, j)
            tempOverlap = blendOverlap(Atemp, Btemp, i)
            tempAverage = np.average(tempOverlap)

            # Finally, compute the |x^2| for the overlap and record the result
            x2[j-yslidemin, i-xslidemin] = np.sqrt(np.average(X*X) / tempAverage)

            # Calculate and display loop progress
            count += 1
            if count % 100 == 0:
                thistime = time.time()
                thisperiod = (thistime - prevtime) / 100 * (numcalcs - count)
                totalperiod = (thistime - starttime) / count * (numcalcs - count)
                estimated_time = (totalperiod + thisperiod) / 2
                print(f'\rCalculating overlaps... '
                      f'Progress: {count / numcalcs:.1%} complete!\t'
                      f'Estimated Time Remaining: '
                      f'{estimated_time/60:.2f} minutes\t',
                      end='\r')
                prevtime = time.time()
    print(f'\rCalculating overlaps... '
          f'Progress: {count / numcalcs:.1%} complete!\t'
          f'Estimated Time Remaining: '
          f'{estimated_time/60:.2f} minutes\t',
          end='\r')
    print(f'\nTotal Time Elapsed: {(time.time() - starttime) / 60:.2f} minutes\n')
    
    # Find the minimum |x^2| value. This is interpreted as the best match
    minidx = np.unravel_index(np.argmin(x2), x2.shape)
    imin = minidx[1] + xslidemin
    jmin = minidx[0] + yslidemin
    print(f'Optimal overlap at B-coordinate ({imin}, {jmin})')
    
    # Time to construct the panorama image
    Rtemp = A.copy()
    S = B.copy()
    print(f'Starting Image Shapes: {A.shape}, {B.shape}')
    
    # Trim off the extra rows due to the y-shift
    Rtemp, S = trimRows(Rtemp, S, jmin)
    print(Rtemp.shape, S.shape)
    
    # Start with the left side of image A, which is not overlapped
    R = Rtemp[:, :-imin]
    print(R.shape, S.shape)
    
    # Blend the overlapping region according to a weighted average,
    # column by column.
    R = np.hstack((R, blendOverlap(Rtemp, S, imin)))
    print(R.shape, S.shape)
    
    # Append the remainder of the right image to the end of the result
    R = np.hstack((R, S[:, imin:]))
    print(f'Final Result shape: {R.shape}\n')
    
    if plotStats:
        # Plot the averaged column intensity of the result
        plt.ion()
        plt.figure(1)
        plt.title('Average Column Intensity of Result')
        plt.plot(np.average(R, 0),
                 label=f'{plotIndex}')
        plt.legend()
        plt.show()
        plt.pause(1)
    
        # Plot the averaged column intensity difference b/w result and original
        plt.figure(2)
        plt.title(
            'Average Column Intensity Difference from Original (Right Image)')
        plt.plot(np.average(R[:, -B.shape[1]:], 0) - np.average(B, 0),
                 label=f'{plotIndex}')
        plt.legend()
        plt.show()
        plt.pause(1)
        
        # Plot the |x^2| value vs the shift index (2D implot)
        fig = plt.figure(3)
        plt.suptitle('Overlap-Normalized RMS Difference Map')
        ax = fig.add_subplot(totalPlots, 1, plotIndex)
        ax.imshow(x2, cmap='jet', extent=[xslidemin, xslidemax,
                                          yslidemax, yslidemin], aspect='auto')
        plt.show()
        plt.pause(1)
        
        # Plot the image difference between the result and the original
        fig = plt.figure(4)
        plt.suptitle('Result vs Original (Right Image) Difference Map')
        ax = fig.add_subplot(totalPlots, 1, plotIndex)
        ax.imshow(R[:, -S.shape[1]:] - S, cmap='bwr', vmin=-25, vmax=25, aspect='auto')
        
        plt.show()
        plt.pause(1)

    print('Overlap Computations Complete!')
    return R, jmin // 2


def blendOverlap(A, B, i):
    xa = np.linspace(1, 0, i)
    xb = np.linspace(0, 1, i)
    y = np.ones((A.shape[0], 1))
    blenderA = xa * y
    blenderB = xb * y
    R = A[:, -i:] * blenderA + B[:, :i] * blenderB
    return R


def trimRows(A, B, j):
    # This method assumes that the height of image A is equal or less than
    # the height of image B.
    ah = A.shape[0]
    bh = B.shape[0]
    #print(f'Trimming {j} rows... ', A.shape, B.shape)
    if j < 0:
        # Trim off the bottom "j" rows of image A
        # and the top "j" rows of image B.
        if j + bh - ah >= 0:
            pass
        else:
            A = A[:j+(bh-ah), :]
        B = B[-j:-j+ah, :]
    
    elif j > 0:
        # Trim off the top "j" rows of image A
        # and the bottom "j" rows of image B.
        A = A[j:, :]
        B = B[:ah-j, :]
    
    else:
        # No vertical shift, just ensure the image shapes match
        B = B[:ah, :]
        
    #print(f'New shapes: {A.shape}, {B.shape}')
    return A, B


def main(pathWithAsterisk,
         plotStats=False, plotResult=False,
         resize=False, resize_height=300):
    # files must be named such that the left image has a lower alphanumeric
    # name than the right image, and thus the left image will be passed as
    # imageA, and the right image will be passed as imageB.
    files = glob.glob(pathWithAsterisk)
    
    # load the two images (and optionally resize them for faster processing)
    imageA = cv2.imread(files[0])
    imageB = cv2.imread(files[1])
    if resize:
        imageA = imutils.resize(imageA, height=resize_height)
        imageB = imutils.resize(imageB, height=resize_height)
    ymin = 0
    ymax = imageA.shape[0]
    
    # Set up the overlap shift plot
    #plt.figure(3)
            
    # Send the images to the shift and blend algorithm and log the result
    fileCount = 1
    result, prevYShift = shiftAndBlend(imageA, imageB, 0,
                                       fileCount, len(files)-1,
                                       plotStats=plotStats)
    if prevYShift > 0:
        ymin += prevYShift
    else:
        ymax += prevYShift
    fileCount += 1
    print(f'Number of files stitched: {fileCount} of {len(files)}\n')
    
    # If there are more than 2 files, loop over the entire file list and
    # keep appending the next image on the right side of the previous result
    if len(files) > 2:
        for f in files[2:]:
            print(f'Stitching filename {f}')
            imageB = cv2.imread(f)
            if resize:
                imageB = imutils.resize(imageB, height=resize_height)
            result, prevYShift = shiftAndBlend(result, imageB, prevYShift,
                                               fileCount, len(files)-1,
                                               plotStats=plotStats)
            if prevYShift > 0:
                ymin += prevYShift
            else:
                ymax += prevYShift
            fileCount += 1
            print(f'Number of files stitched: {fileCount} of {len(files)}\n')
            
    # Convert the result back into a uint8 array
    result = result.astype('uint8')

    if plotStats:
        # Plot results
        fig = plt.figure(4)
        axes = fig.get_axes()
        norm = mpl.colors.Normalize(vmin=-25, vmax=25)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=axes)
    
    if plotResult:
        plt.figure()
        plt.suptitle('Panorama vs Original Comparison')
        ax = plt.subplot(2, 1, 1)
        ax.imshow(result, cmap='jet', vmax=255, vmin=0, aspect='auto')
        plotCount = len(files) + 1
        for f in files:
            original = cv2.imread(f)
            origgray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            ax = plt.subplot(2, len(files), plotCount)
            ax.imshow(origgray, cmap='jet', vmax=255, vmin=0, aspect='auto')
            plotCount += 1

    return result, (ymin, ymax)


def compareResult(original, result, ybounds, resize_height):
    origTemp = imutils.resize(original, height=resize_height)
    ymin, ymax = ybounds
    origTrim = origTemp[ymin:ymax, :]
    origResize = cv2.resize(origTrim, (result.shape[1], result.shape[0]))
    print(origResize.shape, origTrim.shape, origTemp.shape, original.shape, result.shape)
    r = result.astype('int')
    o = origResize.astype('int')
    rmsError = np.sqrt((r - o)**2)
    rmsTotal = np.average(rmsError)
    fig, axes = plt.subplots(6)
    axes[0].imshow(result, cmap='jet', aspect='auto')
    axes[1].imshow(original, cmap='jet', aspect='auto')
    axes[2].imshow(origTemp, cmap='jet', aspect='auto')
    axes[3].imshow(origTrim, cmap='jet', aspect='auto')
    axes[4].imshow(origResize, cmap='jet', aspect='auto')
    axes[5].imshow(r-o, cmap='jet', aspect='auto')
    return rmsTotal
    
    
def blur(orig, ksize):
    return cv2.blur(orig, (ksize, ksize))


def gaussianBlur(orig, ksize):
    return cv2.GaussianBlur(orig, (ksize, ksize), 0)
    

def splitTest(origPath, noiseMag, width=300):
    for f in glob.glob('*.png'):
        os.remove(f)
    orig = cv2.imread(origPath)
    gray = toGray(orig)
    noise = gray + noiseMag * np.random.normal(size=gray.shape)
    noise = np.where(noise < 0, 0, noise)
    noise = np.where(noise >=255, 255, noise)
    
    splitResults = split(noise, width)
    for i in range(len(splitResults)):
        cv2.imwrite(f'{chr(65+i)}.png', splitResults[i])


def split(img, width):
    temp = img.copy()
    images = []
    random.seed()
    while temp.shape[1] > width:
        splitIndex = int(width / 2 + width / 3 * random.random())
        images.append(temp[:, :width])
        temp = temp[:, width-splitIndex:]
    images.append(temp)
    return images
