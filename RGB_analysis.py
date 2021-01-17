#!/usr/bin/env python
import os
import cv2
import matplotlib.pyplot as plt
import argparse
from plantcv import plantcv as pcv

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 230) 

### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug",
                        help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
                        default=None)
    args = parser.parse_args()
    return args

def plot_hotspots(directory, filename):
 
    # Read image
    in_full_path  = os.path.join(directory, filename)
    outfile = 'Hotspot' + filename 
    out_full_path = os.path.join(directory, outfile)
    
    img, path, filename = pcv.readimage(in_full_path, mode="rgb")
   
    # Convert RGB to HSV and extract the saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    # Threshold the saturation image
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, max_value=255, object_type='light')

    # Median Blur
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
     
    edge = cv2.Canny(s_mblur, 60, 180)
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.imshow(edge, cmap='Greys')     
       
    contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    centroids = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, cnt in enumerate(contours):                 
           if (cv2.contourArea(cnt) > 10 ):
               moment = cv2.moments(contours[i]) 
               Cx = int(moment["m10"]/moment["m00"])
               Cy = int(moment["m01"]/moment["m00"])
               center = (Cx, Cy)
               centroids.append((contours, center, moment["m00"], 0))
               cv2.circle(img, (Cx, Cy), 5, (255, 255, 255), -1)
               coordinate = '(' + str(Cx) + ',' + str(Cy) + ')'
               cv2.putText(img, coordinate, (Cx,Cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
               print(cv2.contourArea(cnt),Cx, Cy)
    cv2.imwrite(out_full_path, img)                    
    #cv2.imshow('canvasOutput', img);    
    #cv2.waitKey(0)
    
    return


def main():
    # Get options
    args = options()
    
    files = ('SI20S1-01720image1.jpg', 'SI20S1-01720image2.bmp' , 'SI20S1-01720image3.bmp',
    'SI20S1-01891image1.jpg', 'SI20S1-01891image2.bmp' , 'SI20S1-01891image3.bmp',
    'SI20S1-01984image1.bmp', 'SI20S1-01984image2.bmp' , 'SI20S1-01984image3.jpg',
    'SI20S1-01985image1.bmp', 'SI20S1-01985image2.bmp' , 'SI20S1-01985image3.jpg',
    'SI20S1-02244image1.bmp', 'SI20S1-02244image2.bmp' , 'SI20S1-02244image3.bmp',
    'SI20S1-02361image1.bmp', 'SI20S1-02361image2.bmp' , 'SI20S1-02361image3.bmp',
    'SI20S1-02361image4.bmp', 'SI20S1-02361image5.bmp' , 'SI20S1-02361image6.bmp')
   
    directory = 'c:/projects/202010'
    for i, filename in enumerate(files):   
        plot_hotspots(directory, filename)
    
    return
   
if __name__ == '__main__':
    main()
