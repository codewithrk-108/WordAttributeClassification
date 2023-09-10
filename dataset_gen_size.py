import numpy as np
from PIL import Image
import os
from multiprocessing import Pool, cpu_count
import cv2

def resize_aspect(orignal_image):
    w = orignal_image.shape[1]//3
    h = orignal_image.shape[0]//3
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

def aspect_conservative_resize(orignal_image,height=170,width=1200):
    w = int(width)
    h = int(orignal_image.shape[0]*(width/orignal_image.shape[1]))
    if h>height:
        w=int(height*orignal_image.shape[1]/orignal_image.shape[0])
        h = height
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

def aspect_conservative_resize_height(orignal_image,height=170,width=1200):
    h = int(height)
    w = int(orignal_image.shape[1]*(height/orignal_image.shape[0]))
    if w>width:
        h=int(width*orignal_image.shape[0]/orignal_image.shape[1])
        w = width
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

def centralizer(orignal_image,height=170,width=1200):
    pre_processed_image = orignal_image
    if orignal_image.shape[1]>width:
        pre_processed_image = aspect_conservative_resize(orignal_image,height,width)
    
    elif orignal_image.shape[0]>height:
        pre_processed_image = aspect_conservative_resize_height(orignal_image,height,width)
    
    # print(orignal_image.shape)
    plain_image = np.zeros((height,width),dtype=np.float32)
    plain_image.fill(255)
    # print(np.unique(plain_image))
    width_centering_factor = (plain_image.shape[1] - pre_processed_image.shape[1])//2
    height_centering_factor = (plain_image.shape[0] - pre_processed_image.shape[0])//2  
    plain_image[height_centering_factor:pre_processed_image.shape[0]+height_centering_factor,width_centering_factor:pre_processed_image.shape[1]+width_centering_factor] = pre_processed_image[:,:]

    return plain_image
