"""
Author : Rohan Kumar
Date : 11-12 June 2023


Description : Font Attribute Detection Model (Tesseract) using TesserOCR
Tech Used : Tesseract Library and TesserOCR library which is python wrapper 
of the former library.

Tested Document images on this pipeline codebase.

"""
import h5py
import os
import torch
import numpy as np
import json
from scipy import misc
# Let's pick the desired backend
# os.environ['USE_TF'] = '1'
os.environ['USE_TORCH'] = '1'
import matplotlib.pyplot as plt
from doctr.models import ocr_predictor
from collections import OrderedDict
from doctr.io import DocumentFile
import cv2
import json

from PIL import Image, ImageFile
import random
from keras import backend as K

from keras.models import load_model

# Let's pick the desired backend
# os.environ['USE_TF'] = '1'
os.environ['USE_TORCH'] = '1'

Folder = ['test_corpus']

# Specify Directory and Lang Folder
directory = '..'
directory_new = "output"

#Loading the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
predictor = ocr_predictor(pretrained=True).to(device)

# original saved file with DataParallel
state_dict = torch.load("/scratch/db_resnet50.pt")


#COMMENT THE BELOW LINES OF CODE IF DataParallel was not used during training (provided weights used DataParallel)
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

    
    
predictor.det_predictor.model.load_state_dict(new_state_dict)

def doctr_predictions(directory):
    #Gets the predictions from the model
    
    doc = DocumentFile.from_images(directory)
    result = predictor(doc)
    dic = result.export()
    
    page_dims = [page['dimensions'] for page in dic['pages']]
    
    regions = []
    abs_coords = []
    
    regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
    abs_coords = [
        [[int(round(word['geometry'][0][0] * dims[1])), 
        int(round(word['geometry'][0][1] * dims[0])), 
        int(round(word['geometry'][1][0] * dims[1])), 
        int(round(word['geometry'][1][1] * dims[0]))] for word in words]
        for words, dims in zip(regions, page_dims)
    ]
    return abs_coords

def aspect_conservative_resize(orignal_image,width=1200):
    w = int(width)
    h = int(orignal_image.shape[0]*(width/orignal_image.shape[1]))
    if h>170:
        w=int(170*orignal_image.shape[1]/orignal_image.shape[0])
        h = 170
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_CUBIC)

def aspect_conservative_resize_height(orignal_image,height=170):
    h = int(height)
    w = int(orignal_image.shape[1]*(height/orignal_image.shape[0]))
    if w>1200:
        h=int(1200*orignal_image.shape[0]/orignal_image.shape[1])
        w = 1200
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_CUBIC)

def centralizer(orignal_image,height=170,width=1200):
    pre_processed_image = orignal_image
    if orignal_image.shape[1]>1200:
        # print("hello")
        pre_processed_image = aspect_conservative_resize(orignal_image,1200)
    
    elif orignal_image.shape[0]>170:
        # print("cool")
        pre_processed_image = aspect_conservative_resize_height(orignal_image,170)
    
    # print(orignal_image.shape)
    plain_image = np.zeros((height,width),dtype=np.float32)
    plain_image.fill(255)
    # print(np.unique(plain_image))
    width_centering_factor = (plain_image.shape[1] - pre_processed_image.shape[1])//2
    height_centering_factor = (plain_image.shape[0] - pre_processed_image.shape[0])//2
    plain_image[height_centering_factor:pre_processed_image.shape[0]+height_centering_factor,width_centering_factor:pre_processed_image.shape[1]+width_centering_factor] = pre_processed_image[:,:]
    return plain_image
# K.set_image_data_format('channels_last')

def modify_im(word):
    # since survey shows 9 max

    h = 130
    w = int((h/word.shape[0])*word.shape[1])
    print(word.shape)
    image = cv2.resize(word,(w,h),interpolation=cv2.INTER_CUBIC)
    return centralizer(image)

# model = load_model('./deep_font.h5')
count=0
cnt=0

# arr = np.zeros((50,50))
# img = Image.fromarray(arr)
# print(img)
json_dict={}
reverse_map={}
# aspect_ratio = 0
bbox_pixels = []
for pos,fold in enumerate(Folder):
    pat = directory+"/"+fold+"/"
    
    for file in os.listdir(pat):
        #loading the image file and running inference
        image_file = directory+"/"+fold+"/"+file
        pred = doctr_predictions(image_file)
        #creating output image
        itr=0
        # image_1 = Image.open(image_file)
        # image_1 = image_1.convert("L")
        # image_1 = np.array(image_1)
        image_1 = cv2.imread(image_file)
        # print(image_1.shape)
        d=0
        json_dict[file]=[]
        for w in pred[0]:
            word = np.array(image_1[w[1]:w[3],w[0]:w[2]])
            word = cv2.cvtColor(word,cv2.COLOR_BGR2GRAY)
            # aspect_ratio = max(aspect_ratio,word.shape[1]/word.shape[0])
            word = modify_im(word)
            print(word.shape)
            
            bbox_pixels.append(word)
            # cropped_word_im = np.array(generate_crop(word_im,50,5))
            cropped_word_im = np.array([word])
            
            # print(cropped_word_im)
            if(cropped_word_im.shape[0]>0):
                json_dict[file].append({})
                json_dict[file][len(json_dict[file])-1]['bb_dim'] = [w[0],w[1],w[2],w[3]]
                json_dict[file][len(json_dict[file])-1]['bb_ids'] = []
                # cropped_word_im = cropped_word_im.reshape((cropped_word_im.shape[0],cropped_word_im.shape[1],cropped_word_im.shape[2],1))
                for im_ind in range(cropped_word_im.shape[0]):
                      json_dict[file][len(json_dict[file])-1]['bb_ids'].append({})
                      json_dict[file][len(json_dict[file])-1]['bb_ids'][len(json_dict[file][len(json_dict[file])-1]['bb_ids'])-1]['id'] = count
                      json_dict[file][len(json_dict[file])-1]['bb_ids'][len(json_dict[file][len(json_dict[file])-1]['bb_ids'])-1]['attb'] = {
                            'isBold':False,
                            'isItalic':False
                      }
                      reverse_map[str(count)]={}
                      reverse_map[str(count)]['file'] = file
                      reverse_map[str(count)]['index'] = len(json_dict[file])-1

                     # plt.imshow(cropped_word_im[im_ind],cmap='gray')
                     # plt.savefig('./cool.png')
                      count+=1


with open('/scratch/bbox_info.json','w') as file:
      json.dump(json_dict,file)

with open('/scratch/rev_map.json','w') as file:
      json.dump(reverse_map,file)

with h5py.File('/scratch/bbox_pixels' + '.hdf5', 'w') as f:
	f.create_dataset('bbox_pixels',data=bbox_pixels)
