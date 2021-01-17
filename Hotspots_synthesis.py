# Author: Dr. DING Zhongqiang
#
# By downloading, copying, installing or using the software you agree to this license. 
# If you do not agree to this license, do not download, install, copy or use the software.
#
# Copyright (C) 2000-2020, Intel Corporation, all rights reserved.
# Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
# Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
# Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
# Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
# Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
# Copyright (C) 2019-2020, Xperience AI, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#

#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.io as io

RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
BLUE   = (255, 0, 0)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 230) 

Debug = False

def syn_hotspots(directory, file_fre, file_phase):

    in_full_path  = os.path.join(directory, file_fre)     
    magnitudeimg  = io.imread(in_full_path)   
    mag_red       = magnitudeimg[:, :, 0]
    mag_green     = magnitudeimg[:, :, 1]  
    mag_blue      = magnitudeimg[:, :, 2]    
    
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.imshow(magnitudeimg)
    plt.show()
    
    in_full_path   = os.path.join(directory, file_phase)     
    phaseimg       = io.imread(in_full_path)   
    phase_red   = phaseimg[:, :, 0]
    phase_green = phaseimg[:, :, 1]  
    phase_blue  = phaseimg[:, :, 2]    
    
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.imshow(phaseimg)
    plt.show()    

    result_array = np.zeros_like(magnitudeimg)
    
    synsig_red_complex    = mag_red*np.exp(1j*phase_red)
    synsig_red            = np.fft.ifft(synsig_red_complex).real
    synsig_green_complex  = mag_red*np.exp(1j*phase_green)
    synsig_green          = np.fft.ifft(synsig_green_complex).real
    synsig_blue_complex   = mag_blue*np.exp(1j*phase_blue)
    synsig_blue           = np.fft.ifft(synsig_blue_complex).real
    result_array[:, :, 0] = synsig_red
    result_array[:, :, 1] = synsig_green
    result_array[:, :, 2] = synsig_blue       
     
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.imshow(result_array)   
    plt.show()
    
    ##############################

    amplitude_array = np.zeros_like(magnitudeimg)
    phase_array     = np.zeros_like(phaseimg)

    img_red         = np.fft.fft(synsig_red_complex)
    phase_red       = np.angle(img_red)
    magnitude_red   = 20*np.log(np.abs(img_red))
    amplitude_array[:, :, 0] = magnitude_red
    phase_array[:,:,0]       = phase_red
    
    img_green       = np.fft.fft(synsig_green_complex)
    phase_green     = np.angle(img_green)
    magnitude_green = 20*np.log(np.abs(img_green))
    amplitude_array[:, :, 1] = magnitude_green
    phase_array[:,:,1]       = phase_green    
    
    img_blue        = np.fft.fft(synsig_blue_complex)
    phase_blue      = np.angle(img_blue)
    magnitude_blue  = 20*np.log(np.abs(img_blue))
    amplitude_array[:, :, 2] = magnitude_blue
    phase_array[:,:,2]       = phase_blue    
    
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.imshow(amplitude_array)   
    plt.show()
    
    fig, ax = plt.subplots(1, figsize=(12,8))    
    plt.imshow(phase_array)  
    plt.show()        

    return


def main():        
        
    directory = 'c:/projects/images/thermal'
   
    filename_frq   = '001_amplitude.jpg'
    filename_phase = '001_phase.jpg'
    syn_hotspots(directory, filename_frq, filename_phase )
    
    #filename_frq   = '002_amplitude.jpg'
    #filename_phase = '002_phase.jpg'
    #syn_hotspots(directory, filename_frq, filename_phase )    
    
    '''
    img = cv2.imread('messi5.jpg',0)

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    '''
    
    return
   
if __name__ == '__main__':
    main()