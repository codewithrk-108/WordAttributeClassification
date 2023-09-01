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

def resize_image(image, image_dimension):
	""" Input: Image Path
		Output: Image
		Resizes image to height of 105px, while maintaining aspect ratio
	"""
	base_height = image_dimension
	img = image
	height_percent = (base_height/float(img.size[1]))
	wsize = int((float(img.size[0])*float(height_percent)))
	# print("Width", wsize)
	img = img.resize((wsize, base_height),Image.ANTIALIAS )

	return img

def generate_crop(img, image_dimension, num_vals):
	""" Input: Image object, the width and height of our image, number of cropped images
		Output: A list of mnormalized numpy arrays normalized between 0 and 1.
		Randomly generates 15 cropped images.
	"""
	cropped_images = []
	width = len(np.array(img)[1])

	if width > image_dimension + num_vals:
		bounds = random.sample(range(0, width-image_dimension), num_vals)
		for i in range(num_vals):
			# left top right bottom
			new_img = img.crop((bounds[i], 0, bounds[i] + image_dimension, image_dimension))
			new_img = np.array(new_img)/255.0

			cropped_images.append(new_img)
	return cropped_images

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
        image_1 = Image.open(image_file)
        image_1 = image_1.convert("L")
        image_1 = np.array(image_1)
        # print(image_1.shape)
        d=0
        json_dict[file]=[]
        for w in pred[0]:
            word = np.array(image_1[w[1]:w[3],w[0]:w[2]])
            word_pil = Image.fromarray(word)
            word_im=resize_image(word_pil, 50)
          
            cropped_word_im = np.array(generate_crop(word_im,50,5))
            
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

                      bbox_pixels.append(cropped_word_im[im_ind])
                      count+=1

with open('bbox_info.json','w') as file:
      json.dump(json_dict,file)

with open('rev_map.json','w') as file:
      json.dump(reverse_map,file)

with h5py.File('bbox_pixels' + '.hdf5', 'w') as f:
	f.create_dataset('bbox_pixels',data=bbox_pixels)
          
