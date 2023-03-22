"""
copy and paste these functions wherever you like inside your nb
remember to import the following libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
"""

#image plotter
def rgbshow(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
#image comparer
def compare(i1,i2,vert=False):
    h=1 if not vert else 2
    v=2 if not vert else 1
    if not vert:
        plt.figure(figsize=(40,20), dpi=50)
    else:
        plt.figure(figsize=(16,8))
    
    plt.subplot(h,v,1)
    if vert:
        plt.xticks([])
    rgbshow(i1)
    
    plt.subplot(h,v,2)
    if not vert:
        plt.yticks([])
    rgbshow(i2)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()