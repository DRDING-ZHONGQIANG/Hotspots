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
import math
import imutils
from   plantcv import plantcv as pcv
from   scipy   import ndimage
from   PIL     import Image 

RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
BLUE   = (255, 0, 0)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 230) 
BLACK  = (0, 0, 0)
LIGHT_GRAY    = (195, 195, 195)
Debug = False

def plot_optical(directory, filename, pattern_list):
 
    # Read image
    in_full_path  = os.path.join(directory, filename)     
    img, path, filename = pcv.readimage(in_full_path, mode="rgb")
    img_optical = img.copy()    
    
    # converting to gray scale
    gray = cv2.cvtColor(img_optical, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    # convolute with proper kernels
    ddepth      = cv2.CV_32F
    kernel_size = 3
    laplacian = cv2.Laplacian(img,ddepth,kernel_size)
    #laplacian     = cv2.convertScaleAbs(laplacian_dst)
    
    sobelx    = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely    = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y     
    absx      = cv2.convertScaleAbs(sobelx)
    absy      = cv2.convertScaleAbs(sobely)
    edge      = cv2.addWeighted(absx, 0.5, absy, 0.5,0)         
    canny = cv2.Canny(img, 20, 16)      
    
    '''
    outfile = 'Laplacian_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, laplacian) 
    
    outfile = 'Sobel_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, edge) 
    '''
    
    outfile = 'Canny_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, canny) 
  
    '''
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([]) 
    
    plt.subplot(2,2,2),plt.imshow(laplacian)
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
 
    plt.subplot(2,2,3),plt.imshow(edge,cmap = 'gray')
    plt.title('Sobel'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(2,2,4),plt.imshow(canny,cmap = 'gray')
    plt.title('Canny'), plt.xticks([]), plt.yticks([])    
    '''  
    '''
    rho             = 1  # distance resolution in pixels of the Hough grid
    theta           = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold       = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap    = 20 # maximum gap in pixels between connectable line segments
    line_image      = np.copy(canny) * 0 # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments    
    lines           = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]),
                       min_line_length, max_line_gap)   
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x1 != x2) and (y1 != y2) :
                length = math.sqrt((y2-y1)**2+(x2-x1)**2)
                slope = abs(((y2-y1)/(x2-x1)))
                if (slope > 1) and (slope < 2 )  and (length > 10):    
                    cv2.line(canny,(x1,y1),(x2,y2),BLUE,1)    
    outfile = 'Line_' + filename 
    out_full_path = os.path.join(directory, outfile)
    cv2.imwrite(out_full_path, canny) 
    '''    
    found = False
    for i, pattern_filename in enumerate(pattern_list): 
        template = cv2.imread(pattern_filename)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) 
        (height, width) = template.shape[::-1]
        match = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5    
        loc = np.where( match >= threshold)    
        for pt in zip(*loc[::-1]):
               w, h = template.shape[::-1]
               cv2.rectangle(canny, pt, (pt[0] + 10, pt[1] + 10), WHITE, 1) 
               found = True
        
    if found == True:
        outfile = 'Pattern_' + filename 
        out_full_path = os.path.join(directory, outfile)
        cv2.imwrite(out_full_path, canny) 
    
    plt.show()  
       
    return    


def main():
        
    if Debug == False:  
        '''
        directory = 'c:/projects/202010'
        files = ('Gray_org_SI20S1-01720image2.bmp' , 'Gray_org_SI20S1-01720image3.bmp',
        'Gray_org_SI20S1-01891image2.bmp' , 'Gray_org_SI20S1-01891image3.bmp',
        'Gray_org_SI20S1-01984image1.bmp', 'Gray_org_SI20S1-01984image2.bmp' ,
        'Gray_org_SI20S1-01985image1.bmp', 'Gray_org_SI20S1-01985image2.bmp' , 
        'Gray_org_SI20S1-02244image1.bmp', 'Gray_org_SI20S1-02244image2.bmp' , 
        'Gray_org_SI20S1-02361image1.bmp', 'Gray_org_SI20S1-02361image2.bmp' , 
        'Gray_org_SI20S1-02361image3.bmp', 'Gray_org_SI20S1-02361image4.bmp', 
        'Gray_org_SI20S1-02361image5.bmp', 'Gray_org_SI20S1-02361image6.bmp')
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
       files = ('SI16A1-00280 M2673B PG-LQFP-144 image1.jpg','Gray_org_SI15A1-01135 M2673B PG-LQFP-144 image1.jpg','Gray_org_SI20S1-02244image3.bmp','Gray_org_SI20S1-02244image2.bmp','Gray_org_SI20S1-02244image1.bmp')  
          
    pattern_list = ('wire_pattern1.jpg', 'wire_pattern2.jpg', 'wire_pattern3.jpg', 'wire_pattern4.jpg', 'wire_pattern5.jpg','wire_pattern6.jpg','wire_pattern7.jpg', 'wire_pattern8.jpg')
    for i, filename in enumerate(files):   
        plot_optical(directory, filename, pattern_list)
           
    return
   
if __name__ == '__main__':
    main()