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
GRAY   = (128, 128, 128)

Debug = True

def plot_hotspots(directory, filename,convex_hull_flag):
 
    # Read image
    in_full_path  = os.path.join(directory, filename)     
    img, path, filename = pcv.readimage(in_full_path, mode="rgb")
    img_thermal = img.copy()
    
    # Convert RGB to HSV and extract the saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    # Threshold the saturation image
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, max_value=255, object_type='light')

    # Median Blur
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)     
    edge = cv2.Canny(s_mblur, 60, 180)
      
    outfile = 'Gray_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, edge)         
   
    # Contours extraction       
    contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    hull_list = []
    if (convex_hull_flag == True):
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)  
        contours = hull_list
    
    mask = np.zeros(edge.shape, np.uint8)
    for i, cnt in enumerate(contours):  
           cv2.drawContours(mask, contours, i,255, cv2.FILLED)   
    thermal = thermal_image(img_thermal, mask)  
    optical = optical_image(img_thermal, thermal)

    if (convex_hull_flag == True):
        outfile = 'Thermal_CH_' + filename
    else:
        outfile = 'Thermal_' + filename            
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, thermal)   
        
    if (convex_hull_flag == True):
        outfile = 'Optical_CH_' + filename
    else:
        outfile = 'Optical_' + filename            
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, optical)  
    
    centroids = []
    
    if (convex_hull_flag == True):
        area_threshold = 30
    else:
        area_threshold = 10    
 
    for i, cnt in enumerate(contours):  
           cv2.drawContours(mask, contours, i,255, cv2.FILLED)                      
           if (cv2.contourArea(cnt) > area_threshold ):
               moment = cv2.moments(contours[i]) 
               Cx = int(moment["m10"]/moment["m00"])
               Cy = int(moment["m01"]/moment["m00"])
               center = (Cx, Cy)
               centroids.append((contours, center, moment["m00"], 0))
               #cv2.circle(img, (Cx, Cy), 5, (255, 255, 255), -1)
               coordinate = '(' + str(Cx) + ',' + str(Cy) + ')'
               cv2.putText(img, coordinate, (Cx,Cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
               print(cv2.contourArea(cnt),Cx, Cy)     
    
    if Debug == True:
        fig, ax = plt.subplots(1, figsize=(12,8))    
        plt.imshow(mask, cmap='Greys') 
    
    if (convex_hull_flag == True):    
        outfile = 'Hotspots_CH_' + filename 
    else:
        outfile = 'Hotspots_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, img)                 
        
    return

def thermal_image(org, mask):    
    thermal = cv2.bitwise_and(org, org,mask = mask)   
    return  thermal

def optical_image(org, thermal):
    optical = org - thermal
    for i in range(thermal.shape[0]):
        for j in range(thermal.shape[1]):
             if ( (thermal[i,j][0] != 0 ) or (thermal[i,j][1] != 0) or (thermal[i,j] [2] !=0)):
                 optical[i,j][0] = GRAY[0]
                 optical[i,j][1] = GRAY[1]
                 optical[i,j][2] = GRAY[2]    
    return optical

def main():
        
    if Debug == False:    
        files = ('SI20S1-01720image1.jpg', 'SI20S1-01720image2.bmp' , 'SI20S1-01720image3.bmp',
        'SI20S1-01891image1.jpg', 'SI20S1-01891image2.bmp' , 'SI20S1-01891image3.bmp',
        'SI20S1-01984image1.bmp', 'SI20S1-01984image2.bmp' , 'SI20S1-01984image3.jpg',
        'SI20S1-01985image1.bmp', 'SI20S1-01985image2.bmp' , 'SI20S1-01985image3.jpg',
        'SI20S1-02244image1.bmp', 'SI20S1-02244image2.bmp' , 'SI20S1-02244image3.bmp',
        'SI20S1-02361image1.bmp', 'SI20S1-02361image2.bmp' , 'SI20S1-02361image3.bmp',
        'SI20S1-02361image4.bmp', 'SI20S1-02361image5.bmp' , 'SI20S1-02361image6.bmp')
    else:
       files = ('SI20S1-02361image1.bmp','SI20S1-02361image2.bmp')  
    
    directory = 'c:/projects/202010'
    for i, filename in enumerate(files):
        convex_hull_flag = False  
        plot_hotspots(directory, filename, convex_hull_flag)
        
        convex_hull_flag = True  
        plot_hotspots(directory, filename, convex_hull_flag)
    
    return
   
if __name__ == '__main__':
    main()
