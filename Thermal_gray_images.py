"""
Created on Fri Nov 13 18:18:58 2020

@author: HP
"""

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
import argparse
from   plantcv import plantcv as pcv

RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
BLUE   = (255, 0, 0)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 230) 
BLACK  = (0, 0, 0)
LIGHT_GRAY    = (165, 165, 165)
Debug = False

def optical_xray_images(directory, filename): 
    # Read image
    in_full_path  = os.path.join(directory, filename)     
    img, path, filename = pcv.readimage(in_full_path, mode="rgb")
    img_thermal = img.copy()
    
    # record origial gray image
    optical_org,pos = optical_image_org(img_thermal)
    optical_org     = smooth(optical_org, pos)
    outfile = 'Gray_org_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, optical_org)       
    return

def smooth(org, pos):
    optical = org.copy()
    x_limit = org.shape[0]
    y_limit = org.shape[1]
    for i, item in enumerate(pos):
        i = item[0]
        j = item[1]        
        org[i,j][0] = median_choose(org, i, j, 9, x_limit,y_limit) 
        org[i,j][1] = org[i,j][0]                            
        org[i,j][2] = org[i,j][0]       
    return optical

def median_choose(org, pixel_x, pixel_y, size, x_limit,y_limit):     
    
    matrix_size = int((size - 1)/2)
    pixels = []
    for j in range(matrix_size):
          if (pixel_y - j) > 0:
                pixels.append(org[pixel_x,pixel_y-j][0])
          if ((pixel_y + j ) < y_limit): 
                pixels.append(org[pixel_x,pixel_y+j][0])     
    for i in range(matrix_size):
        for j in range(matrix_size):      
            if ((pixel_x -i) > 0) and (pixel_y - j) > 0:
                pixels.append(org[pixel_x-i,pixel_y-j][0])
            if ((pixel_x +i) < x_limit) and ((pixel_y +j ) < y_limit): 
                pixels.append(org[pixel_x+i,pixel_y+j][0])                 
    return med(pixels)

def med(points):    
    points = np.sort(points, axis = 0)    
    n = len(points) 
    return points[n//2] 

def optical_image(org, thermal):
    optical = org - thermal
    for i in range(thermal.shape[0]):
        for j in range(thermal.shape[1]):
             if ( (thermal[i,j][0] != 0 ) or (thermal[i,j][1] != 0) or (thermal[i,j] [2] !=0)):
                 optical[i,j][0] = LIGHT_GRAY[0]
                 optical[i,j][1] = LIGHT_GRAY[1]
                 optical[i,j][2] = LIGHT_GRAY[2]    
    return optical

def optical_image_org(org):
    pos=[]
    for i in range(org.shape[0]):
        for j in range(org.shape[1]):
                 if pixel_in_GRAY_COLORMAP(org[i,j]) == False: 
                     org[i,j][0] = LIGHT_GRAY[0]
                     org[i,j][1] = LIGHT_GRAY[1]
                     org[i,j][2] = LIGHT_GRAY[2] 
                     pos.append((i,j))
    return org, pos

def pixel_in_GRAY_COLORMAP(pixel):    
    if (pixel[0] == WHITE[0]) and (pixel[1] == WHITE[1]) and (pixel[2] == WHITE[2]):
        return False        
    if (pixel[0] == pixel[1]) and (pixel[1] == pixel[2]):
        return True 
    
    delta1 = int(abs(int(pixel[0])-int(pixel[1])))
    delta2 = int(abs(int(pixel[0])-int(pixel[2])))
    delta3 = int(abs(int(pixel[1])-int(pixel[2])))
    low_threshold = 1
    high_threshold =10
    
    if (low_threshold < delta1 < high_threshold) and (low_threshold < delta2 < high_threshold) and (low_threshold < delta3 < high_threshold):
       return True
   
    return False

def main():
        
    if Debug == False: 
        '''
        directory = 'c:/projects/202010'
        files = ('SI20S1-01720image2.bmp' , 'SI20S1-01720image3.bmp',
        'SI20S1-01891image2.bmp' , 'SI20S1-01891image3.bmp',
        'SI20S1-01984image1.bmp', 'SI20S1-01984image2.bmp' , 
        'SI20S1-01985image1.bmp', 'SI20S1-01985image2.bmp' ,
        'SI20S1-02244image1.bmp', 'SI20S1-02244image2.bmp' , 'SI20S1-02244image3.bmp',
        'SI20S1-02361image1.bmp', 'SI20S1-02361image2.bmp' , 'SI20S1-02361image3.bmp',
        'SI20S1-02361image4.bmp', 'SI20S1-02361image5.bmp' , 'SI20S1-02361image6.bmp')
        '''
        directory = 'c:/projects/202011/package'
        files  = ('SI15A1-01135 M2673B PG-LQFP-144 image1.jpg',
                 'SI16A1-00280 M2673B PG-LQFP-144 image1.jpg',
                 'SI16A1-00281 M2673B PG-LQFP-144 image1.jpg','SI16A1-00636 M2617A PG-TQFP-100 image1.jpg',
                 'SI16A1-00636 M2617A PG-TQFP-100 image2.jpg','SI16A1-00730 M1747C PG-BGA-416 image1.png',
                 'SI16A1-00730 M1747C PG-BGA-416 image2.png','SI16A1-00730 M1747C PG-BGA-416 image3.jpg',
                 'SI17A1-00332 M1947B PG-LQFP-176 image1.png','SI17A1-01026 M2682B PG-LQFP-144 image1.png',
                 'SI17A1-01026 M2682B PG-LQFP-144 image2.jpg',
                 'SI19A1-00410 S7189K PG-VQFN-48 image1.png', 'SI19A1-00484 M2662C PG-LQFP-100 image3.jpg')
                 #'SI19A1-00484 M2662C PG-LQFP-100 image1.png','SI19A1-00484 M2662C PG-LQFP-100 image2.png')        
             
    else:      
       directory = 'c:/projects/202010'
       files = ('SI20S1-02244image2.bmp', 'SI20S1-02244image3.bmp')   
      
   
    for i, filename in enumerate(files):
        optical_xray_images(directory, filename)     
    
    return
   
if __name__ == '__main__':
    main()