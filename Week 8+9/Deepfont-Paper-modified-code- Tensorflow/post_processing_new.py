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
directory = '.'
directory_new = "output"

#Loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def glorify_edges(image, low_threshold=100, high_threshold=200):
    # Convert the image color(BGR) to gray
    # print(image.shape)
    image = image.astype('uint8')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # # Convert edges back to BGR for combination
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # # Combine original image with edges
    glorified = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)

    return glorified

def augment_and_patch(image, patch_size=(50,50),flag=1):

    if flag==1:
    # 1. Add Gaussian Noise
        image = util.random_noise(image, mode='gaussian', mean=0, var=(3/255)**2)

    # 2. Add Gaussian Blur
    # kernel_size = random.randint(2, 3)
        kernel_size = 3
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # 3. Perspective Rotation
        random_scale = 0.1 * np.random.randn(2) + 1
        random_translation = 0.8 * np.random.randn(2)
        transform_param = AffineTransform(scale=random_scale, translation=random_translation)
        image = transform.warp(image, transform_param)

    # 4. Add Shading
    # Generate gradient
        gradient = np.linspace(0.5, 1, image.shape[1])
        image = (image.transpose(2,0,1) * gradient).transpose(1,2,0)
        image = image*255
        image = image.astype('uint8')

    # glorify edgesimage = image.astype('uint8')
	# image = image
        image = glorify_edges(image)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = image/255
    # image = image.astype('float64')

    # Resize keeping aspect ratio to height 50
    height, width = image.shape[:2]
    new_height = 50
    new_width = int((new_height / height) * width)
    image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

    # # Extract 50 x 50 patches
    patches = []
    for i in range(0, image.shape[1], patch_size[0]):
    #     for j in range(0, image.shape[1], patch_size[1]):
            patch = image[:, i:i+patch_size[0]]
            if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                patches.append(patch)

    return patches

K.set_image_data_format('channels_last')

# model = load_model('./deep_font.h5')
count=0
cnt=0

# arr = np.zeros((50,50))
# img = Image.fromarray(arr)
# print(img)
json_dict={}
reverse_map={}
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
            word = np.array(image_1[w[1]:w[3],w[0]:w[2],:])
          
            # cropped_word_im = np.array(generate_crop(word_im,50,5))
            cropped_word_im = np.array(augment_and_patch(word,flag=0))
            
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
                            'isSemiBold':False,
                            'isItalic':False
                      }
                      reverse_map[str(count)]={}
                      reverse_map[str(count)]['file'] = file
                      reverse_map[str(count)]['index'] = len(json_dict[file])-1

                     # plt.imshow(cropped_word_im[im_ind],cmap='gray')
                     # plt.savefig('./cool.png')
                      bbox_pixels.append(cropped_word_im[im_ind])
                      count+=1

with open('bbox_info.json','w') as file:
      json.dump(json_dict,file)

with open('rev_map.json','w') as file:
      json.dump(reverse_map,file)

with h5py.File('bbox_pixels' + '.hdf5', 'w') as f:
	f.create_dataset('bbox_pixels',data=bbox_pixels)
          
