"""
Author : Rohan Kumar
Date : 23 June 2023


Description : Letter Morphology Voting Model (Modified)
Link of the Paper Referenced: https://arxiv.org/pdf/2205.07683.pdf 

(Did not Consider Letter segmentation and directly skeletonized words)

"""


from PIL import Image
import cv2
import numpy as np
import io
import os
import sys
from collections import OrderedDict
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import torch
import tesserocr
import time

from skimage.morphology import skeletonize,medial_axis
from skimage.util import invert

Folder = ['English']
Languages = ['eng']

# Specify Directory and Lang Folder
directory = 'Languages'
directory_new = "Lang_output"

#Loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = ocr_predictor(pretrained=True).to(device)

# original saved file with DataParallel
state_dict = torch.load("../db_resnet50.pt",map_location=torch.device('cpu'))

#COMMENT THE BELOW LINES OF CODE IF DataParallel was not used during training (provided weights used DataParallel)
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

# Find mean boldness Function for letter
def find_mean_boldness(thinned,dist_trans):
    """
    thinned - medial axis image
    dist_trans - distance transform matrix 
    obtained by applying medial axis transform. 
    """
    sum=0
    num_pix_med=0
    for i in range(thinned.shape[0]):
        for j in range(thinned.shape[1]):
            if(thinned[i][j]):
                sum+=dist_trans[i][j]
                num_pix_med+=1
    if num_pix_med==0:
        return 0
    sum = round(sum/num_pix_med,3)
    return sum #mean
    
    
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

"""
MODEL CODE 

START
"""

for pos,fold in enumerate(Folder):
    pat = directory+"/"+fold+"/"
    # Path
    
    for file in os.listdir(pat):
        #loading the image file and running inference
        image_file = directory+"/"+fold+"/"+file
        pred = doctr_predictions(image_file)

        # Reading image as numpy array : image_np
        image_np = cv2.imread(image_file)

        # Testing and Accuracy Phase
        # psm = 8 for reading word wise images

        # Customize the tessdata path
        with tesserocr.PyTessBaseAPI(path='/home/rohan/CVIT/Doctr/tesseract/tessdata',lang=Languages[pos],psm=8) as api:
            
            # Reading a PIX object from Pillow library
            img_pix = Image.open(directory+"/"+fold+"/"+file)

            # Creating duplicates numpy arrays of image
            img = np.array(img_pix)
            image = np.array(img_pix)

            # Collection of all words with mean thickness
            theta_image=[]


            for itr,w in enumerate(pred[0]):

                # extracting word
                img_word = np.array(img[w[1]:w[3],w[0]:w[2]])
                cv2.imwrite('../word.png',img_word)
                
                # converting to an Pix object 
                word_img = Image.open('../word.png')

                """
                Image Preprocessing

                START
                """
                word_img = cv2.cvtColor(word_img,cv2.COLOR_BGR2GRAY)
                word_img = cv2.adaptiveThreshold(word_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
                kernel = np.array([
                    [1,1,1],
                    [1,1,1],
                    [1,1,1],
                ],dtype='uint8')
                word_img = cv2.morphologyEx(word_img,cv2.MORPH_ERODE,kernel,iterations=1)
                word_img = cv2.morphologyEx(word_img,cv2.MORPH_DILATE,kernel,iterations=3)
                """
                END
                """

                # duplicate the word image
                word_img_dup = np.copy(word_img)

                # apply medial axis transform of the scikit - image library
                # return binary Dtype
                word_img,distance = medial_axis(word_img,return_distance=True)

                # Find the mean thickness of the word (Custom function)
                mean = find_mean_boldness(word_img,distance)

                # Pushing word mean thickness in theta_image
                theta_image.append(mean)
            
            # Voting
            theta_image = np.array(theta_image)

            # Classification array for words
            Classified_words = np.zeros(theta_image.shape)

            # Hyperparameter/Threshold for bold classification
            alpha=0.2
            for index in range(theta_image.shape[0]):
                if theta_image[index] > np.median(theta_image)+ alpha*np.std(theta_image):
                    Classified_words[index]=1
                else:
                    Classified_words[index]=0
            
            # Annotation to display the results
            for it,w in enumerate(pred[0]):
                if(Classified_words[it]):
                    cv2.rectangle(image_np,(w[0],w[1]),(w[2],w[3]),(0,0,255),2)
                else:
                    cv2.rectangle(image_np,(w[0],w[1]),(w[2],w[3]),(0,255,0),1)

            # Saving the imagein respective language folder in {directory_new} folder.
            cv2.imwrite(directory_new+'/'+fold+'/'+'out_'+file,image_np)

"""
END
"""