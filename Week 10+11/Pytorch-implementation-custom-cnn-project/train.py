import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import gc
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool,cpu_count
import os
import numpy as np
from PIL import Image
from create_train_set_labels import font_dict
from torchvision.transforms import ToTensor
import h5py
import time

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

import glob
from torch.utils.data import Dataset, DataLoader

from albumentations import (
    Compose, 
    GaussNoise,
    GaussianBlur,
    Perspective,
    ShiftScaleRotate,  
    IAAAffine,
    RandomBrightnessContrast
)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="CNNArch",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "ADOBE_VFR",
    "epochs": 20,
    }
)


parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-m', type=str, required=True, help='id parameter')
args = parser.parse_args()


# Define the CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(7*50*128, 128)  # After three pooling layers, dimensions are reduced to (7, 50)
        self.fc2 = nn.Linear(128, 3)          # Three output nodes
        
        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv + ReLU + Pool for layer 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv + ReLU + Pool for layer 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Conv + ReLU + Pool for layer 3
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 7*50*128)
        
        # Fully connected + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Fully connected + Sigmoid
        x = torch.sigmoid(self.fc2(x))
        
        return x


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

# Load the dataset
# dataset = ImageFolder(root='path_to_dataset_folder', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# target_image = cv2.imread('./aug.png')
# tim_2 = np.random.randint(127, 256, [170, 1200, 3], dtype=np.uint8)
# tim2 = tim2.astype('uint8')
# print(target_image.shape)


class CustomDataset(Dataset):
    def __init__(self,X,Y):
        self.train_inputs = X
        self.train_labels = Y
    
    def __len__(self):
        return len(self.train_inputs)
    
    def __getitem__(self, index):

        img_path = self.train_inputs[index]
        # print(img_path[1:])
        label = self.train_labels[index]

        # start_t = time.time()
        link = img_path.decode()       
        # print(link)
        # link = link.split('/')[-1:-4:-1]
        # link = f'/scratch/{link[2]}/{link[1]}/{link[0]}'
        img = np.array(Image.open(link))
        img = np.reshape(img,(img.shape[0],img.shape[1],1))
        # print(img.shape)
        # print(img)
        # print(img.shape)
        transform=ToTensor()
        return transform(img),label
        

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
    
    gray_image = cv2.cvtColor(augmented['image'],cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./cool.png',gray_image)
    transform = ToTensor()
    return transform(gray_image)

# batch_size = 300

def get_data(train_filename,train_lbl_filename,val_filename,val_lbl_filename):
    with h5py.File(train_filename, 'r') as hf:
        name = train_filename.split('/')[-1].split('.')[-2]
        train_inputs = hf[name][:]
    with h5py.File(train_lbl_filename, 'r') as hf:
        name = train_lbl_filename.split('/')[-1].split('.')[-2]
        train_labels = hf[name][:]
    with h5py.File(val_filename, 'r') as hf:
        name = val_filename.split('/')[-1].split('.')[-2]
        val_inputs = hf[name][:]
    with h5py.File(val_lbl_filename, 'r') as hf:
        name = val_lbl_filename.split('/')[-1].split('.')[-2]
        val_labels = hf[name][:]
    
    return train_inputs,train_labels,val_inputs,val_labels

def get_data_seg(train_noneinp_filename,train_nonelbl_filename,train_boldinp_filename,train_boldlbl_filename,train_italicinp_filename,train_italiclbl_filename,train_bolditalicinp_filename,train_bolditaliclbl_filename,val_filename,val_lbl_filename):
    with h5py.File(train_noneinp_filename, 'r') as hf:
        name = train_noneinp_filename.split('/')[-1].split('.')[-2]
        train_noneinp_inputs = hf[name][:]
    with h5py.File(train_nonelbl_filename, 'r') as hf:
        name = train_nonelbl_filename.split('/')[-1].split('.')[-2]
        train_nonelbl_inputs = hf[name][:]
    with h5py.File(train_boldinp_filename, 'r') as hf:
        name = train_boldinp_filename.split('/')[-1].split('.')[-2]
        train_boldinp_inputs = hf[name][:]
    with h5py.File(train_boldlbl_filename, 'r') as hf:
        name = train_boldlbl_filename.split('/')[-1].split('.')[-2]
        train_boldlbl_inputs = hf[name][:]
    with h5py.File(train_italicinp_filename, 'r') as hf:
        name = train_italicinp_filename.split('/')[-1].split('.')[-2]
        train_italicinp_inputs = hf[name][:]
    with h5py.File(train_italiclbl_filename, 'r') as hf:
        name = train_italiclbl_filename.split('/')[-1].split('.')[-2]
        train_italiclbl_inputs = hf[name][:]
    with h5py.File(train_bolditalicinp_filename, 'r') as hf:
        name = train_bolditalicinp_filename.split('/')[-1].split('.')[-2]
        train_bolditalicinp_inputs = hf[name][:]
    with h5py.File(train_bolditaliclbl_filename, 'r') as hf:
        name = train_bolditaliclbl_filename.split('/')[-1].split('.')[-2]
        train_bolditaliclbl_inputs = hf[name][:]
    
    with h5py.File(val_filename, 'r') as hf:
        name = val_filename.split('/')[-1].split('.')[-2]
        val_inputs = hf[name][:]
    with h5py.File(val_lbl_filename, 'r') as hf:
        name = val_lbl_filename.split('/')[-1].split('.')[-2]
        val_labels = hf[name][:]
    
    return train_noneinp_inputs,train_nonelbl_inputs,train_boldinp_inputs,train_boldlbl_inputs,train_italicinp_inputs,train_italiclbl_inputs,train_bolditalicinp_inputs,train_bolditaliclbl_inputs,val_inputs,val_labels


def get_data_seg_real(train_noneinp_filename,train_nonelbl_filename,train_boldinp_filename,train_boldlbl_filename,train_italicinp_filename,train_italiclbl_filename,train_bolditalicinp_filename,train_bolditaliclbl_filename,val_filename,val_lbl_filename):
    with h5py.File(train_noneinp_filename, 'r') as hf:
        name = train_noneinp_filename.split('/')[-1].split('.')[-2]
        paths = hf[name][:]
        train_noneinp_inputs=[]
        for i in paths:
            train_noneinp_inputs.append(cv2.imread(i.decode()))
    with h5py.File(train_nonelbl_filename, 'r') as hf:
        name = train_nonelbl_filename.split('/')[-1].split('.')[-2]
        train_nonelbl_inputs = hf[name][:]

    with h5py.File(train_boldinp_filename, 'r') as hf:
        name = train_boldinp_filename.split('/')[-1].split('.')[-2]
        paths = hf[name][:]
        train_boldinp_inputs=[]
        for i in paths:
            train_boldinp_inputs.append(cv2.imread(i.decode()))

    with h5py.File(train_boldlbl_filename, 'r') as hf:
        name = train_boldlbl_filename.split('/')[-1].split('.')[-2]
        train_boldlbl_inputs = hf[name][:]

    with h5py.File(train_italicinp_filename, 'r') as hf:
        name = train_italicinp_filename.split('/')[-1].split('.')[-2]
        paths = hf[name][:]
        train_italicinp_inputs=[]
        for i in paths:
            train_italicinp_inputs.append(cv2.imread(i.decode()))

    with h5py.File(train_italiclbl_filename, 'r') as hf:
        name = train_italiclbl_filename.split('/')[-1].split('.')[-2]
        train_italiclbl_inputs = hf[name][:]


    with h5py.File(train_bolditalicinp_filename, 'r') as hf:
        name = train_bolditalicinp_filename.split('/')[-1].split('.')[-2]
        paths = hf[name][:]
        train_bolditalicinp_inputs=[]
        for i in paths:
            train_bolditalicinp_inputs.append(cv2.imread(i.decode()))

    with h5py.File(train_bolditaliclbl_filename, 'r') as hf:
        name = train_bolditaliclbl_filename.split('/')[-1].split('.')[-2]
        train_bolditaliclbl_inputs = hf[name][:]

    
    with h5py.File(val_filename, 'r') as hf:
        name = val_filename.split('/')[-1].split('.')[-2]
        val_inputs = hf[name][:]
    with h5py.File(val_lbl_filename, 'r') as hf:
        name = val_lbl_filename.split('/')[-1].split('.')[-2]
        val_labels = hf[name][:]
    
    return train_noneinp_inputs,train_nonelbl_inputs,train_boldinp_inputs,train_boldlbl_inputs,train_italicinp_inputs,train_italiclbl_inputs,train_bolditalicinp_inputs,train_bolditaliclbl_inputs,val_inputs,val_labels


def getdata(root_dir,folders_to_be_excluded,train_num_each_font=650,test_num_each_font=198,validation_num_each_font=150,num_classes=3):
    
    train_none_inputs = []
    train_none_labels=[]
    train_bold_inputs = []
    train_bold_labels = []
    train_italic_inputs = []
    train_italic_labels = []
    train_bolditalic_inputs = []
    train_bolditalic_labels = []
    validation_inputs = []
    validation_labels = []
    test_inputs = []
    test_labels = []
    ct=0 
    gc.collect()
    for font_folder in os.listdir(root_dir):
        if int(font_folder) in folders_to_be_excluded:
            continue
        
        fol_path = os.path.join(root_dir,font_folder)
        # array_img = np.random.choice(os.listdir(fol_path),train_num_each_font+validation_num_each_font+test_num_each_font)
        array_img = os.listdir(fol_path)
        # print(array_img)
        if array_img==[]:
            continue

        ct+=len(array_img)
        # remove semibold
        try:
            font_dict[int(font_folder)][font_dict[int(font_folder)].index(3)]=1
            # print(font_dict[int(font_folder)])
        except:
            pass
        
        for itr,img in enumerate(array_img[:int(len(array_img)*0.65)]): # goes through all font folders
            image_path = fol_path + "/" + img
            lbl = font_dict[int(font_folder)]
            
            if 0 in font_dict[int(font_folder)]:
                train_none_inputs.append(image_path)
            if 1 in font_dict[int(font_folder)] and 2 in font_dict[int(font_folder)]:
                train_bolditalic_inputs.append(image_path)
            elif 1 in font_dict[int(font_folder)]:
                train_bold_inputs.append(image_path)
            elif 2 in font_dict[int(font_folder)]:
                train_italic_inputs.append(image_path)

            temp = np.zeros(num_classes)
            for el in lbl:
                if el==3:
                    temp[1]=1
                else:
                    temp[el]=1

            if 0 in font_dict[int(font_folder)]:
                train_none_labels.append(temp)
            if 1 in font_dict[int(font_folder)] and 2 in font_dict[int(font_folder)]:
                train_bolditalic_labels.append(temp)
            elif 1 in font_dict[int(font_folder)]:
                train_bold_labels.append(temp)
            elif 2 in font_dict[int(font_folder)]:
                train_italic_labels.append(temp)

        # ct+=len(int(len(array_img)*0.65))
                
        for itr,img in enumerate(array_img[int(len(array_img)*0.65):int(len(array_img)*0.85)]): 
            # goes through all font folders
            image_path = fol_path + "/" + img
            lbl = font_dict[int(font_folder)]
            temp = np.zeros(num_classes)
            for el in lbl:
                if el==3:
                    temp[1]=1
                else:
                    temp[el]=1
            validation_labels.append(temp)
            validation_inputs.append(image_path)
        

        for itr,img in enumerate(array_img[int(len(array_img)*0.85):]): 
            image_path = fol_path + "/" + img
            lbl = font_dict[int(font_folder)]
            temp = np.zeros(num_classes)
            for el in lbl:
                if el==3:
                    temp[1]=1
                else:
                    temp[el]=1
            test_labels.append(temp)
            test_inputs.append(image_path)
    
    # print(ct)
    return train_none_inputs,train_none_labels,train_bold_inputs,train_bold_labels,train_italic_inputs,train_italic_labels,train_bolditalic_inputs,train_bolditalic_labels,validation_inputs,validation_labels,test_inputs,test_labels

folders_to_be_excluded = []

def save(inputs, inputs_file_name, labels, labels_file_name):
	""" Input: inputs, desired inputs file name, labels, desired labels file name, group number to shuffle by
		Output: None

		Shuffles the inputs and labels by a shuffle_size and then dumps them into hdf5 files.
	"""

	with h5py.File('/scratch/'+inputs_file_name + '.hdf5', 'w') as f:
		f.create_dataset(inputs_file_name,data=inputs)

	with h5py.File('/scratch/'+labels_file_name + '.hdf5', 'w') as f:
		f.create_dataset(labels_file_name,data=labels)

with open('./exclude_bold.txt','r') as file:
    for string in file:
        string = string.split(',')
        string[0] = string[0].split('[')[-1]
        string[-1] = string[-1].split(']')[-2]
        string = list(map(int,string))
        # print(string)
        folders_to_be_excluded.extend(string)

with open('./exclude_none.txt','r') as file:
    for string in file:
        string = string.split(',')
        string[0] = string[0].split('[')[-1]
        string[-1] = string[-1].split(']')[-2]
        string = list(map(int,string))
        # print(string)
        folders_to_be_excluded.extend(string)

# y = np.array([[[[1,2,3],[10,11,12]]],[[[4,5,6],[13,14,15]]],[[[7,8,9],[16,17,18]]]])
# print(y.shape)
# print(y)
# x = np.moveaxis(y,3,1)
# print(x)
# print(x.shape)
# X_none,Y_none,X_bold,Y_bold,X_italic,Y_italic,X_bolditalic,Y_bolditalic,vx,vy,x,y = getdata('./new_test',folders_to_be_excluded)
X_none,Y_none,X_bold,Y_bold,X_italic,Y_italic,X_bolditalic,Y_bolditalic,vx,vy = get_data_seg('/scratch/train_none_inputs.hdf5','/scratch/train_none_labels.hdf5','/scratch/train_bold_inputs.hdf5','/scratch/train_bold_labels.hdf5','/scratch/train_italic_inputs.hdf5','/scratch/train_italic_labels.hdf5','/scratch/train_bolditalic_inputs.hdf5','/scratch/train_bolditalic_labels.hdf5','/scratch/val_inputs.hdf5','/scratch/val_labels.hdf5')


X_real_none,Y_real_none,X_real_bold,Y_real_bold,X_real_italic,Y_real_italic,X_real_bolditalic,Y_real_bolditalic,real_vx,real_vy = get_data_seg_real('/scratch/real_train_none_inputs.hdf5','/scratch/real_train_none_labels.hdf5','/scratch/real_train_bold_inputs.hdf5','/scratch/real_train_bold_labels.hdf5','/scratch/real_train_italic_inputs.hdf5','/scratch/real_train_italic_labels.hdf5','/scratch/real_train_bolditalic_inputs.hdf5','/scratch/real_train_bolditalic_labels.hdf5','/scratch/real_val_inputs.hdf5','/scratch/real_val_labels.hdf5')

# print(len(X_none),len(X_bold),len(X_italic),len(X_bolditalic),len(Y_none),len(Y_bold),len(Y_italic),len(Y_bolditalic),len(vx),len(vy),len(x),len(y))
# print(X_none[0],Y_none[0])
# print(X_none[1],Y_none[1])

# Given array sizes
sizes = [len(X_none),len(X_bold), len(X_italic), len(X_bolditalic)]

# # Total samples across all arrays
total_samples = sum(sizes)

# Batch information
batch_size = 180
total_batches = (total_samples//batch_size)+1
# print(total_batches)

# Calculate proportions for each array
proportions = [int(round(s/total_samples * batch_size)) for s in sizes]

# for real data

# Batch information for real data
sizes_real = [len(X_real_none),len(X_real_bold), len(X_real_italic), len(X_real_bolditalic)]
total_samples_real = sum(sizes_real)

batch_size_real = 120
total_batches_real = (total_samples_real//batch_size_real)+1


# Calculate proportions for each array for real
proportions_real = [int(round(s/total_samples_real * batch_size_real)) for s in sizes_real]

def convert_image(img_link):
    return cv2.imread(img_link.decode())

def color_2_grayscale(img):
    transform = ToTensor()
    return transform(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

# Custom dataset loader
class CustomLoader:
    
    def __init__(self, arrays,labels,arrays_real,labels_real):
        self.arrays = arrays
        self.arrays_real = arrays_real
        # self.order=0
        self.prev_batch = []
        self.prev_label = []
        self.labels = labels
        self.labels_real = labels_real
        self.indexes = [0 for _ in arrays]  # Initialize starting index for each array
        self.transform = ToTensor()

        self.indexes_real = [0 for _ in arrays_real]  # Initialize starting index for each array

    def __len__(self):
        return total_batches

    def __getitem__(self, idx):
        batch = []
        lbl = []
        # if idx<self.order:
        #     return self.prev_batch,self.prev_label

        # self.order+=1
        for i, arr in enumerate(self.arrays):
            batch_synth = arr[self.indexes[i]: self.indexes[i] + proportions[i]]
            # Get samples from current array based on proportion and update index
            # with Pool(processes=2) as pool:
            b=[]
            for link in batch_synth:
                b.append(cv2.imread(link.decode()))
            batch_synth = b

            b=[]
            for img in batch_synth:
                new = self.transform(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
                b.append(new)
            batch_synth = b
            # batch_synth = map(convert_image, batch_synth)
            # batch_synth = map(color_2_grayscale, batch_synth)

            batch.extend(batch_synth)
            lbl.extend(self.labels[i][self.indexes[i]: self.indexes[i] + proportions[i]])
            self.indexes[i] += proportions[i]
            self.indexes[i] = self.indexes[i]%len(self.arrays[i])


        for i, arr in enumerate(self.arrays_real):
            # Get samples from current array based on proportion and update index            
            batch_real = arr[self.indexes_real[i]: self.indexes_real[i] + proportions_real[i]]
        
            b=[]
            for img in batch_real:
                new = self.transform(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
                b.append(new)
            batch_real = b
           

            batch.extend(batch_real)
            lbl.extend(self.labels_real[i][self.indexes_real[i]: self.indexes_real[i] + proportions_real[i]])
            self.indexes_real[i] += proportions_real[i]
            self.indexes_real[i] = self.indexes_real[i]%len(self.arrays_real[i])
        
        lbl = np.array(lbl)
        lbl = torch.from_numpy(lbl)

        return [torch.stack(batch),lbl]

# Mock data arrays for demonstration
arrays = [X_none,X_bold,X_italic,X_bolditalic]
arrays_real = [X_real_none,X_real_bold,X_real_italic,X_real_bolditalic]


labels = [Y_none,Y_bold,Y_italic,Y_bolditalic]
labels_real = [Y_real_none,Y_real_bold,Y_real_italic,Y_real_bolditalic]

# Create data loader

count=0
ctr=0

# for i in range(total_batches):
    # a,b,c,d=0,0,0,0
    # print(loader[i][0].shape,loader[i][1].shape)
    # for num,lbl in enumerate(loader[i][1]):
    #     name = ((loader[i][0][num].decode()).split('/')[-2])
    #     temp = np.zeros(3)
    #     for el in font_dict[int(name)]:
    #         if el==3:
    #             temp[1]=1
    #         else:
    #             temp[el]=1
    #     if temp[0]==lbl[0] and temp[1]==lbl[1] and temp[2]==lbl[2]:
    #         pass
    #     else:
    #         count+=1

    #     if lbl[0]==1 and lbl[1]==0 and lbl[2]==0:
    #         a+=1
    #     if lbl[0]==0 and lbl[1]==1 and lbl[2]==0:
    #         b+=1
    #     if lbl[0]==0 and lbl[1]==0 and lbl[2]==1:
    #         c+=1
    #     if lbl[0]==0 and lbl[1]==1 and lbl[2]==1:
    #         d+=1

# print(ctr)
# for i in Y_none:
    # if i[0]==0 and i[1]==1 and i[2]==1:
        # pass
    # else:
        # print("hello") 
# print(len(X_none)+len(X_bold)+len(X_italic)+len(X_bolditalic))
# Create DataLoaders

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# print("Images loaded")
# save(x,'real_test_inputs',y,'real_test_labels')
# save(X_none,'real_train_none_inputs',Y_none,'real_train_none_labels')
# save(X_bold,'real_train_bold_inputs',Y_bold,'real_train_bold_labels')
# save(X_italic,'real_train_italic_inputs',Y_italic,'real_train_italic_labels')
# save(X_bolditalic,'real_train_bolditalic_inputs',Y_bolditalic,'real_train_bolditalic_labels')
# save(vx,'real_val_inputs',vy,'real_val_labels')
# print("saved all !",flush=True)

# train_data = CustomDataset(X_none,Y_none,X_bold,Y_bold,X_italic,Y_italic,X_bolditalic,Y_bolditalic)
# train_loader = DataLoader(train_data, batch_size=300,num_workers=2)

train_loader = CustomLoader(arrays,labels,arrays_real,labels_real)


# for i in range(len(train_loader)):
#     print(len(train_loader[i][0]))

val_data = CustomDataset(vx,vy)
val_data_real = CustomDataset(real_vx,real_vy)

val_loader = DataLoader(val_data, batch_size=300,num_workers=2)
val_loader_real = DataLoader(val_data_real, batch_size=300,num_workers=2)

# Let's suppose that 'model' is your model and it is an instance of nn.Module
model = CustomCNN()

# Define a loss function and an optimizer
criterion = nn.BCELoss()  # Change this if needed
optimizer = optim.Adam(model.parameters(),lr=0.001)  # Change this if needed

# Number of training epochs
n_epochs = 50 # Change this if needed

# Device configuration for GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


print(f"Model number : {args.m}")
if torch.cuda.device_count() >= 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
#   dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
# #   print(type(model))
  pretrained = torch.load(f'/scratch/MixedTrain23lakh+{args.m}_model_cnn.pth')
  ste_prev= pretrained.module.state_dict()
  print(list(ste_prev.keys()))
  custom_jugaad = {}
  for key in ste_prev.keys():
      if key.split('.')[-1] == "num_batches_tracked":
          continue
      custom_jugaad["module."+key] = ste_prev[key]
#   print(custom_jugaad.keys())
  model.load_state_dict(custom_jugaad)
#   for param in model.parameters():
#       param.requires_grad = False
model = model.to(device)

# print(model.parameters())
# for param in model:
#     print(param)


sample_trained = 0
# modified
# Training and validation
for epoch in range(14,14+n_epochs):
    model.train()
    running_loss = 0.0
    train_accuracy=0
    for i in range(total_batches):
        inputs,labels = train_loader[i]
        inputs = inputs.type('torch.FloatTensor')
        labels = labels.type('torch.FloatTensor')
        # print(type(inputs))
        sample_trained+=labels.shape[0]
        # inputs = inputs.to(memory_format=torch.contiguous_format)
        inputs = inputs.to(device)
        labels = labels.to(device)
        

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

        # for num,output in enumerate(outputs):
        #         # for none
        #         if output[0]>0.5:
        #             train_accuracy+=labels[num][0]
        #         else:
        #             if output[1]>theta_b and output[2]>theta_i:
        #                 train_accuracy+=(labels[num][1] and labels[num][2])
        #             elif output[1]>theta_b:
        #                 train_accuracy+=labels[num][1]
        #             elif output[2]>theta_i:
        #                 train_accuracy+=labels[num][2]

    # Print every epoch
    print(f"Epoch {epoch+1}, Training Loss: {running_loss / sample_trained}")

    # Validation
    model.eval()
    

    # custom hyperparameters for validation
    lb,ub = 0.2,0.7
    max_val_accuracy=0
    min_loss= 34567890.0

    # for real
    max_val_accuracy_real=0
    min_loss_real= 34567890.0

    hp = (0,0)
    hp_real = (0,0)
    with torch.no_grad():
        for theta_b in np.arange(lb,ub,0.2):
            for theta_i in np.arange(lb,ub,0.2):
                sample_tested=0
                val_accuracy = 0
                # define thresholds
                val_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in val_loader:
                    inputs = inputs.type('torch.FloatTensor')
                    labels = labels.type('torch.FloatTensor')
                    sample_tested+=labels.shape[0]

                    # print(labels.shape[0])

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    total+=1

                    # add custom evaluation
                    # print(theta_b)

                    for num,output in enumerate(outputs):
                        # for none
                        if output[0]>0.5:
                            val_accuracy+=labels[num][0]
                        else:
                            if output[1]>theta_b and output[2]>theta_i:
                                val_accuracy+=(labels[num][1] and labels[num][2])
                            elif output[1]>theta_b:
                                val_accuracy+=labels[num][1]
                            elif output[2]>theta_i:
                                val_accuracy+=labels[num][2]
                
                # print(sample_tested)
                val_accuracy = val_accuracy.item()/sample_tested
                val_loss = val_loss/sample_tested
                
        

                sample_tested_real=0
                val_accuracy_real = 0
                # define thresholds
                val_loss_real = 0.0
                for inputs, labels in val_loader_real:
                    inputs = inputs.type('torch.FloatTensor')
                    labels = labels.type('torch.FloatTensor')
                    sample_tested_real+=labels.shape[0]

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_real += loss.item()

                    total+=1

                    # add custom evaluation
                    # print(theta_b)

                    for num,output in enumerate(outputs):
                        # for none
                        if output[0]>0.5:
                            val_accuracy_real+=labels[num][0]
                        else:
                            if output[1]>theta_b and output[2]>theta_i:
                                val_accuracy_real+=(labels[num][1] and labels[num][2])
                            elif output[1]>theta_b:
                                val_accuracy_real+=labels[num][1]
                            elif output[2]>theta_i:
                                val_accuracy_real+=labels[num][2]
                    
                val_accuracy_real = val_accuracy_real/sample_tested_real
                val_loss_real /= sample_tested_real

                
                # print(sample_tested)
                
                # for synthetic
                if max_val_accuracy<val_accuracy:
                    max_val_accuracy = val_accuracy
                    hp = theta_b,theta_i
                if min_loss > val_loss : 
                    min_loss = val_loss

                # for real
                if max_val_accuracy_real<val_accuracy_real:
                    max_val_accuracy_real = val_accuracy_real
                    hp_real = theta_b,theta_i
                if min_loss_real > val_loss_real: 
                    min_loss_real = val_loss_real

                print(f"{epoch + 1} => MAX_ACCURACY for synth : {max_val_accuracy} | MAX_ACCURACY for real: {max_val_accuracy_real} | Best HP for synthetic : {hp} | Best HP for real : {hp_real} | MIN_LOSS for synth: {min_loss} | MIN_loss for real : {min_loss_real}",flush=True)
    wandb.log({"train_loss":running_loss /sample_trained,"train_accuracy":train_accuracy/sample_trained,"synth_val_loss":min_loss,"synth_val_accuracy":max_val_accuracy,"real_val_loss":min_loss_real,"real_val_accuracy":max_val_accuracy_real})
    print(f"{epoch + 1} : Best HP (based on val_accuracy) : {hp}",flush=True)
    # torch.save(model, f'/scratch/MixedTrain23lakh+{epoch+1}_model_cnn.pth')

wandb.finish()

# Create the model
# img = Image.open('./sample.png')
# img = np.array(img)
# rgb_im = np.zeros((170,1200,3),dtype=np.uint8)
# # print(img.shape)
# plain = centralizer(img)
# rgb_im[:,:,0] = plain[:]
# rgb_im[:,:,1] = plain[:]
# rgb_im[:,:,2] = plain[:]
# print(rgb_im.shape)

# fig,axs = plt.subplots(1,2,figsize=(50,100))10
# axs[0].set_title('Orignal')10
# transform_im = augment_image(rgb_im)
# axs[1].set_title('Augmented')
# axs[1].imshow(transform_im)
# plt.savefig('./out_sample.png')
# transform_im = cv2.cvtColor(transform_im,cv2.COLOR_BGR2GRAY)
# transform_im = np.reshape(transform_im,(1,transform_im.shape[0],transform_im.shape[1],transform_im.shape[2])).astype(np.float32)

# output= model(transform_im)
# print(output)

    
# epochs = 50
# optimizer = optimizers.Adam(lr=0.001)
# model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])


# # print(np.unique(plain))
# # plt.imshow(plain,cmap='gray')
# # plt.show()

# # print(model.summary())
# # model.forward()
 
# # Define the loss function and optimizer
# # criterion = nn.BCELoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# # # Define the warm-up rate scheduler
# # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / 10)

# def generate_augments_and_onfly_images(train_inputs,train_labels):
#     for i in range(0,len(train_inputs),batch_size):
#         array_sample=[]
#         array_label=[]
#         cnt=0
#         while(cnt<batch_size):
#             img = cv2.imread(train_inputs[i+cnt])
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             img = centralizer(img)
#             rgb_im = np.zeros((170,1200,3),dtype=np.uint8)
#             rgb_im[:,:,0] = img[:]
#             rgb_im[:,:,1] = img[:]
#             rgb_im[:,:,2] = img[:]
#             rgb_im = augment_image(rgb_im)
#             array_sample.append(rgb_im)
#             array_label.append(train_labels[i+cnt])
#             cnt+=1
#         yield np.array(array_sample),np.array(array_label)
        
# # Training loop
# model.fit_generator(generate_augments_and_onfly_images(train_inputs=X,train_labels=Y),steps_per_epoch=len(X)//batch_size,epochs=epochs)
