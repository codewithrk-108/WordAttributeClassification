
from albumentations import (
    Compose, 
    GaussNoise,
    GaussianBlur,
    Perspective,
    ShiftScaleRotate,  
    IAAAffine,
    RandomBrightnessContrast
)
import os
import numpy as np
import cv2

from create_train_set_labels import font_dict
import matplotlib.pyplot as plt

from PIL import Image

def augment_image(image):
    # Define the transformation pipeline
    transform = Compose([
        GaussNoise(var_limit=(0, 2),p=0.6),  # Adds Gaussian noise to the image. The var_limit argument controls the standard deviation
        GaussianBlur(blur_limit=(3, 11),sigma_limit=(2.5,3.5),p=0.7),  # Adds Gaussian blur to the image. The blur_limit argument controls the kernel size for blurring
        # Perspective(scale=(0.05, 0.1),keep_size=False,p=1),  # Applies a random four-point perspective transformation to the image
        ShiftScaleRotate(rotate_limit=0, shift_limit=0.08, scale_limit=0.1,p=0.6),  # Rotates the image by a certain angle
        # IAAAffine(shear=(-20, 20),p=0.6),  # Adds affine transformations to the image (like shear transformations)
        RandomBrightnessContrast(brightness_limit=0.5,contrast_limit=0.4,p=0.7),
    ])

    # Apply the transformations
    augmented = transform(image=image)
    return augmented['image']

img = cv2.imread('./1.png')
img = augment_image(img)
plt.imshow(img)
plt.savefig('out.png')

# folders_to_be_excluded = []

# with open('./exclude_bold.txt','r') as file:
#     for string in file:
#         string = string.split(',')
#         string[0] = string[0].split('[')[-1]
#         string[-1] = string[-1].split(']')[-2]
#         string = list(map(int,string))
#         # print(string)
#         folders_to_be_excluded.extend(string)

# with open('./exclude_none.txt','r') as file:
#     for string in file:
#         string = string.split(',')
#         string[0] = string[0].split('[')[-1]
#         string[-1] = string[-1].split(']')[-2]
#         string = list(map(int,string))
#         # print(string)
#         folders_to_be_excluded.extend(string)

# print(folders_to_be_excluded)
# # print(font_dict)

# ct=0
# dic = {"none":0,"bold":0,"italic":0,"b+i":0}
# for fold in os.listdir('./new_test'):
    
#     if int(fold) in folders_to_be_excluded:
#         continue


#     ct+=1
#     if 0 in font_dict[int(fold)]:
#         dic["none"]+=len(os.listdir(os.path.join('./new_test',fold)))
    
#     if 1 in font_dict[int(fold)] and 2 in font_dict[int(fold)]:
#         dic["b+i"]+=len(os.listdir(os.path.join('./new_test',fold)))
#     elif 1 in font_dict[int(fold)]:
#         dic["bold"]+=len(os.listdir(os.path.join('./new_test',fold)))
#     elif 2 in font_dict[int(fold)]:
#         dic["italic"]+=len(os.listdir(os.path.join('./new_test',fold)))
# print(dic)
# print(ct)

# patches = plt.bar(list(dic.keys()),list(dic.values()))
# plt.bar_label(patches)
# plt.savefig('./real_plot')



