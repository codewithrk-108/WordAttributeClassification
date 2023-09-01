"""
Author : Rohan Kumar
Date : 11-12 June 2023


Description : Font Attribute Detection Model (Tesseract) using TesserOCR
Tech Used : Tesseract Library and TesserOCR library which is python wrapper 
of the former library.

Tested Document images on this pipeline codebase.

"""

from PIL import Image
import cv2
import numpy as np
import io
import os
import tesserocr
import sys
import doc
from collections import OrderedDict
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import torch
import time

Folder = ['English','Hindi','Telugu','Tamil','Gujrati']
Languages = ['eng','hin+eng','tel+eng','tam+eng','guj+eng']

# Specify Directory and Lang Folder
directory = 'Languages'
directory_new = "Lang_output"

#Loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = ocr_predictor(pretrained=True).to(device)

# original saved file with DataParallel
state_dict = torch.load("./db_resnet50.pt")

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

for pos,fold in enumerate(Folder):
    pat = directory+"/"+fold+"/"
    
    for file in os.listdir(pat):
        #loading the image file and running inference
        image_file = directory+"/"+fold+"/"+file
        pred = doctr_predictions(image_file)

        #creating output image
        itr=0
        image_1 = cv2.imread(image_file)
        for w in pred[0]:
            word = np.array(image_1[w[1]:w[3],w[0]:w[2]])
            cv2.imwrite(directory+"/"+fold+"/out_"+str(itr)+".png", word)
            itr+=1

        # Testing and Annotation Phase

        """
        Tesseract Based Testing of Font Attributes

        OEM : engine mode = 0 for legacy engine which supports font
        attribute detection.

        PSM : Page segmentation mode = 8 which supports accepting an 
        image as a word image.

        """
        with tesserocr.PyTessBaseAPI(lang=Languages[pos],oem=0,psm=8) as api:
            for counter in range(itr):
                print(".")
                img = Image.open(directory+"/"+fold+"/out_"+str(counter)+".png")
                api.SetImage(img)
                d = api.Recognize()
                iterator = api.GetIterator()
        
                try:
                    attributes = iterator.WordFontAttributes()
                    w = pred[0][counter]
                    word = np.array(image_1[w[1]:w[3],w[0]:w[2]])
                    if(attributes["bold"] and attributes["italic"]):
                        cv2.rectangle(image_1,(w[0],w[1]),(w[2],w[3]),(255,0,255),2)
                    elif(attributes["bold"]==True):
                        cv2.rectangle(image_1,(w[0],w[1]),(w[2],w[3]),(0,0,255),2)
                    elif(attributes["italic"]==True):
                        cv2.rectangle(image_1,(w[0],w[1]),(w[2],w[3]),(255,0,0),2)
                    else:
                        continue
                except:
                    print("Not detected")
                counter+=1

            # Saving the image in {directory_new} in the respective language folder
            cv2.imwrite(directory_new+"/"+Folder[pos]+"/"+fold+"_out_"+file+".png",image_1)
        for fil in os.listdir(pat):
            if(fil[:3]=="out"):
                os.remove(os.path.join(pat,fil))