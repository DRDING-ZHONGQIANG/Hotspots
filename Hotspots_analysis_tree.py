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
import Thermal_gray_images as tgi
from   plantcv import plantcv as pcv

RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
BLUE   = (255, 0, 0)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 230) 
BLACK  = (0, 0, 0)
LIGHT_GRAY    = (195, 195, 195)

Debug = True
Debug_Wire_Region = True
mag_array =[]

def plot_hotspots(directory, filename,convex_hull_flag):
 
    # Read image
    in_full_path  = os.path.join(directory, filename)     
    img, path, filename = pcv.readimage(in_full_path, mode="rgb")
    img_thermal = img.copy()    
      
    # Convert RGB to HSV and extract the saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    # Threshold the saturation image
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, max_value=255, object_type='light')
    if  Debug == True:
         outfile = 'thresh_' + filename 
         out_full_path = os.path.join(directory, outfile)
         cv2.imwrite(out_full_path, s_thresh)

    # Median Blur
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)     
    if  Debug == True:
         outfile = 'median_filter_' + filename 
         out_full_path = os.path.join(directory, outfile)
         cv2.imwrite(out_full_path, s_mblur) 
    
    edge = cv2.Canny(s_mblur, 60, 180)    
    outfile = 'Gray_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, edge)      
       
    # Contours extraction       
    contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(contours) == 0:
        return
    
    hull_list = []
    if (convex_hull_flag == True):
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)  
        contours = hull_list
    
    mask  = np.zeros(edge.shape, np.uint8)
    hiers = hierarchy[0]
    for i in range(len(contours)):  
            if hiers[i][3] != -1:
               continue
            cv2.drawContours(mask, contours, i,255, cv2.LINE_AA)   
            ## Find all inner contours and draw 
            ch = hiers[i][2]
            while ch != -1:
                cv2.drawContours(mask, contours, ch, (255,0,255), -1, cv2.LINE_AA)
                ch = hiers[ch][0]
           
    thermal = thermal_image(img_thermal, mask) 

    if (convex_hull_flag == True):
        outfile = 'Thermal_tree_CH_' + filename
    else:
        outfile = 'Thermal_tree_' + filename            
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, thermal)  
            
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
               
               if Debug_Wire_Region == True: 
                   
                   grayfile      = 'Gray_org_' + filename 
                   gray_path     = os.path.join(directory, grayfile)
                   gray_img      = cv2.imread(gray_path)
                   gray_process  = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
                   
                   mag = fft_mag(gray_process, Cx, Cy)
                   if ((mag != np.nan) and (mag != np.inf) and (mag != -np.inf)):
                       mag_array.append(mag)  
                
               coordinate = '(' + str(Cx) + ',' + str(Cy) + ')'               
               cv2.putText(img, coordinate, (Cx,Cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)   
               print(cv2.contourArea(cnt),Cx, Cy)     
    
    if Debug_Wire_Region == True: 
        if (len(mag_array) > 0):
            magnitude = sum(mag_array)/len(mag_array)
            text ="MAGNITUDE = " + str(magnitude)
            cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA) 
            print(text)
    
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

def pixel_in_GRAY_COLORMAP(pixel):
    
    if (pixel[0] == WHITE[0]) and (pixel[1] == WHITE[1]) and (pixel[2] == WHITE[2]):
        return False
        
    if (pixel[0] == pixel[1]) and (pixel[1] == pixel[2]):
        return True
     
    return False

def fft_mag(img, Cx, Cy):    
    
    height = img.shape[1];  width = img.shape[0]    
    offset = 15; matrix = np.zeros((2*offset+1,2*offset+1))
    
    matrix[offset][offset] = img[Cx][Cy]
    for j in range(offset):
        if (Cy-j) >= 0:
            matrix[offset][offset-j]  = img[Cx][Cy-j]
        if (Cy+j) <= width:
            matrix[offset][offset+j]  = img[Cx][Cy+j]            
  
    for i  in range(offset):                   
           for j in range(offset):
               if (Cy-j) >= 0:
                   matrix[offset-i][offset-j] = img[Cx-i][Cy-j] 
                   
               if (Cy+j) <= width:
                   matrix[offset-i][offset+j] = img[Cx-i][Cy+j]  
            
           for j in range(offset): 
               if (Cy-j) >= 0:
                   matrix[offset+i][offset-j] = img[Cx+i][Cy-j] 
                   
               if (Cy+j) <= width: 
                   matrix[offset+i][offset-j]  = img[Cx+i][Cy-j]                   
    
    f = np.fft.fft2(matrix)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    magnitude = np.mean(magnitude_spectrum)

    return magnitude


def main():
        
    if Debug == False:  
        '''
        directory = 'c:/projects/202010'
        files = ('SI20S1-01720image1.jpg', 'SI20S1-01720image2.bmp' , 'SI20S1-01720image3.bmp',
        'SI20S1-01891image1.jpg', 'SI20S1-01891image2.bmp' , 'SI20S1-01891image3.bmp',
        'SI20S1-01984image1.bmp', 'SI20S1-01984image2.bmp' , 'SI20S1-01984image3.jpg',
        'SI20S1-01985image1.bmp', 'SI20S1-01985image2.bmp' , 'SI20S1-01985image3.jpg',
        'SI20S1-02244image1.bmp', 'SI20S1-02244image2.bmp' , 'SI20S1-02244image3.bmp',
        'SI20S1-02361image1.bmp', 'SI20S1-02361image2.bmp' , 'SI20S1-02361image3.bmp',
        'SI20S1-02361image4.bmp', 'SI20S1-02361image5.bmp' , 'SI20S1-02361image6.bmp')
        '''        
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
          
        '''
     
        current_dir = 'C:\\projects\\Die_level_data'  
        for r, directory, files in os.walk(current_dir):
            for filename in files:         
                    if ("Gray" not in filename) and ("Hotspot" not in filename) and ("Therm" not in filename):
                        convex_hull_flag = True                   
                        print(r, filename)                       
                        plot_hotspots(r, filename, convex_hull_flag)        
              
    else:
    
       '''
       directory = 'c:/projects/202010' 
       files = ('SI16A1-00280 M2673B PG-LQFP-144 image1.jpg','SI15A1-01135 M2673B PG-LQFP-144 image1.jpg','SI20S1-01720image2.bmp','SI20S1-01720image3.bmp', 'SI20S1-01891image2.bmp','SI20S1-01891image3.bmp'  )  
       '''
     
       '''
       current_dir = 'C:\\projects\\abnormal_cases'  
       for r, directory, files in os.walk(current_dir):
            for filename in files:         
                    if ("Gray" not in filename) and ("Hotspot" not in filename) and ("Therm" not in filename):
                        convex_hull_flag = True                   
                        print(r, filename)
                        plot_hotspots(r, filename, convex_hull_flag)                        
       '''             
    
       '''
          for i, filename in enumerate(files):
            convex_hull_flag = False  
            plot_hotspots(directory, filename, convex_hull_flag)
            
            convex_hull_flag = True  
            plot_hotspots(directory, filename, convex_hull_flag)
       '''  
       
       current_dir = 'C:\\projects\\magnitude_spetrum'  
       for r, directory, files in os.walk(current_dir):
            for filename in files:         
                    if ("Gray" not in filename) and ("Hotspot" not in filename) and ("Therm" not in filename):
                        convex_hull_flag = True                   
                        print(r, filename)
                        tgi.optical_xray_images(r, filename)
                        plot_hotspots(r, filename, convex_hull_flag) 
    return
   
if __name__ == '__main__':
    main()