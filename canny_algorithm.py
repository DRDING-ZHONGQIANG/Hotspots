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

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import codecs
from   scipy import ndimage
import math

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def convolution(img, kernel, average=False):
    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    image_row, image_col = img.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(img.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output

def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D

def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(image, kernel, average=True)

def sobel_edge_detection(img, filter, convert_to_degree=False):
    new_image_x = convolution(img, filter)
    new_image_y = convolution(img, np.flip(filter.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_direction = np.arctan2(new_image_y, new_image_x)
    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180
    return gradient_magnitude, gradient_direction

def canny_step1_Gaussian_smoothing(img, size):
    
    dst = gaussian_blur(img, kernel_size= size)     
    return dst  

def canny_step2_gradient_magnitude(img):
    
    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_magnitude, gradient_direction = sobel_edge_detection(img, edge_filter, convert_to_degree=True)
    return gradient_magnitude

def canny_step2_gradient_direction(img):
    
    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_magnitude, gradient_direction = sobel_edge_detection(img, edge_filter, convert_to_degree=True)    
    return gradient_direction 

def canny_step3_non_maximum_superssion(gradient_magnitude, gradient_direction):
    
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]
            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]
            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]
            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
                
    return output


def canny_step4_hysteresis(img, weak):    
    img_row, img_col = img.shape
    top_to_bottom = img.copy()
    for row in range(1, img_row):
        for col in range(1, img_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0
    bottom_to_top = img.copy()
    for row in range(img_row - 1, 0, -1):
        for col in range(img_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0
    right_to_left = img.copy()
    for row in range(1, img_row):
        for col in range(img_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0
    left_to_right = img.copy()
    for row in range(img_row - 1, 0, -1):
        for col in range(1, img_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0
    output = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    output[output > 255] = 255       
    return output


def canny_step4_threshold(img, low, high, weak):
    
    output = np.zeros(img.shape)
    strong = 255
    strong_row, strong_col = np.where(img >= high)
    weak_row, weak_col     = np.where((img <= high) & (img >= low))
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col]     = weak
    
    return output


def extraction(path, filename, filename_suffix, scalar, l_threshold, h_threshold): 
       
    infile = os.path.join(path, filename + "." + filename_suffix)      
    with codecs.open(infile, encoding='utf-8-sig') as f:
        data = np.loadtxt(f)    
        
    filename_suffix = 'jpg'    
    outfile = os.path.join(path, filename + "." + filename_suffix) 
    data = data * scalar
    cv2.imwrite(outfile, data)     
    
    infile = os.path.join(path, filename + "." + filename_suffix)    
    img    = cv2.imread(infile,0) 
   
    step1  = canny_step1_Gaussian_smoothing(img, 21)    
    outfile = os.path.join(path, filename + "_step1." + filename_suffix)
    cv2.imwrite(outfile, step1)    
      
    step2_m = canny_step2_gradient_magnitude(step1)    
    step2_d = canny_step2_gradient_direction(step1) 
    outfile = os.path.join(path, filename + "_step2." + filename_suffix)
    cv2.imwrite(outfile,  step2_m)  
     
    step3   = canny_step3_non_maximum_superssion(step2_m, step2_d)
    outfile  = os.path.join(path, filename + "_step3." + filename_suffix)
    cv2.imwrite(outfile, step3)
    
    weak = 50
    new_image = canny_step4_threshold(step3, l_threshold, h_threshold, weak=weak)     
    step4     = canny_step4_hysteresis(new_image, weak)
    outfile  = os.path.join(path, filename + "_step4." + filename_suffix)
    cv2.imwrite(outfile, step4)     
    return
    
def batch(path):    

    filename_suffix = 'asc'     
    for filename in os.listdir(path):
        
       if 'ampl' in filename.lower() and '.asc' in filename.lower():    
          filename = filename.replace('.asc','')  
          extraction(path, filename, filename_suffix, 256, 5, 40)
    
       if 'phase' in filename.lower() and '.asc' in filename.lower():
           filename = filename.replace('.asc','') 
           extraction(path, filename, filename_suffix, 1, 5, 20)
    
    return

def main():   
    
    '''
    path = 'C:\\projects\\IR_data' 
    filename_suffix = 'asc' 
    filename = 'unit4_0_028V_15mA_100s_Ampl'
    extraction(path, filename, filename_suffix, 256, 5, 40)
    filename = 'unit4_0_028V_15mA_100s_phase'
    extraction(path, filename, filename_suffix, 1, 5, 20)
    
    path = 'C:\\projects\\images\\Different Durations\\reject_asc' 
    batch(path)
    '''
    
    path = 'C:\\projects\\images\\Different Durations\\good_asc' 
    batch(path)
    
    
    
    return
    
if __name__ == '__main__':
    main()    