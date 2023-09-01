import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import gc
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import h5py
import time
from create_train_set_labels import font_dict

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
# wandb.init(
#      # set the wandb project where this run will be logged
#      project="CNNArch",
    
#      # track hyperparameters and run metadata
#      config={
#      "learning_rate": 0.001,
#      "architecture": "CNN",
#      "dataset": "ADOBE_VFR",
#      "epochs": 20,
#      }
#  )


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
        img = img/255
        # print(img.shape)
        # print(img)
        # print(img.shape)
        transform=ToTensor()
        return transform(img),label
        

# def augment_image(image):
#     # Define the transformation pipeline
#     transform = Compose([
#         GaussNoise(var_limit=(0, 2),p=0.6),  # Adds Gaussian noise to the image. The var_limit argument controls the standard deviation
#         GaussianBlur(blur_limit=(3, 7),sigma_limit=(2.5,3.5),p=0.7),  # Adds Gaussian blur to the image. The blur_limit argument controls the kernel size for blurring
#         # Perspective(scale=(0.05, 0.1),keep_size=False,p=1),  # Applies a random four-point perspective transformation to the image
#         # ShiftScaleRotate(rotate_limit=0, shift_limit=0.08, scale_limit=0.1,p=0.6),  # Rotates the image by a certain angle
#         # IAAAffine(shear=(-20, 20),p=0.6),  # Adds affine transformations to the image (like shear transformations)
#         # RandomBrightnessContrast(brightness_limit=0.8,contrast_limit=0.4,p=0.6),
#     ])

#     # Apply the transformations
#     augmented = transform(image=image)
#     return augmented['image']


def getdata(root_dir,folders_to_be_excluded,num_classes=3):
    
    train_inputs = []
    train_labels = []
    validation_inputs = []
    validation_labels = []
    test_inputs = []
    test_labels = []
    ct=0 
    gc.collect()
    for font_folder in os.listdir(root_dir):
        # print("err")
        if int(font_folder) in folders_to_be_excluded:
            continue
        
        # print("no")
        fol_path = os.path.join(root_dir,font_folder)
        # print(fol_path)
        array_img = os.listdir(fol_path)
        ct+=len(array_img)

        for itr,img in enumerate(array_img[:int(len(array_img)*(0.6))]): # goes through all font folders
            image_path = fol_path + "/" + img
            lbl = font_dict[int(font_folder)]
            
            train_inputs.append(image_path)
            temp = np.zeros(num_classes)
            for el in lbl:
                if el==3:
                    temp[1]=1
                else:
                    temp[el]=1
            train_labels.append(temp)
                
        for itr,img in enumerate(array_img[int(len(array_img)*(0.6)):int(len(array_img)*(0.85))]): 
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
        

        for itr,img in enumerate(array_img[int(len(array_img)*(0.85)):int(len(array_img))]): 
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
    
    print(ct)
    return train_inputs,train_labels,validation_inputs,validation_labels,test_inputs,test_labels


batch_size = 100

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

X,Y,vx,vy = get_data('/scratch/real_train_inputs.hdf5','/scratch/real_train_labels.hdf5','/scratch/real_val_inputs.hdf5','/scratch/real_val_labels.hdf5')
print(len(X),len(vy))
# # X,Y,vx,vy,x,y = getdata('./new_test',folders_to_be_excluded)

# # save(x,'real_test_inputs',y,'real_test_labels')
# # save(X,'real_train_inputs',Y,'real_train_labels')
# # save(vx,'real_val_inputs',vy,'real_val_labels')
# # print("saved all !",flush=True)

# # print(len(X),len(Y),X[0],len(vx),len(vy),vx[0],vy[0])
# train_data = CustomDataset(X,Y)
# train_loader = DataLoader(train_data, batch_size=100,num_workers=2)
# val_data = CustomDataset(vx,vy)
# val_loader = DataLoader(val_data, batch_size=100,num_workers=2)

# # Let's suppose that 'model' is your model and it is an instance of nn.Module
# model = CustomCNN()

# # Define a loss function and an optimizer
# criterion = nn.BCELoss()  # Change this if needed
# optimizer = optim.Adam(model.parameters(),lr=0.002)  # Change this if needed
# # changing the learning rate to twice
# # Number of training epochs
# n_epochs = 200  # Change this if needed

# #Device configuration for GPU if available
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

# if torch.cuda.device_count() >= 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)
#   pretrained = torch.load('/scratch/23lakh+52_model_cnn.pth')
#   ste_prev= pretrained.module.state_dict()
#   custom_jugaad = {}
#   for key in ste_prev.keys():
#       if key.split('.')[-1] == "num_batches_tracked":
#           continue
#       print(key,ste_prev[key].requires_grad)
#       custom_jugaad["module."+key] = ste_prev[key]
#   model.load_state_dict(custom_jugaad)
  

# # for param in model.parameters():
#     # param.requires_grad = False

# #print(model.fc.parameters().shape)
# # Unfreeze the fc layer
# # for param in model.fc.parameters():
#     # param.requires_grad = True
# model = model.to(device)

# # Training and validation
# for epoch in range(0,n_epochs):
#     model.train()
    
#     #fine-tuning
#     ctor = 0
#     for param in model.parameters():
#         if ctor<6:
#         	param.requires_grad = False
#         ctor+=1
     
#     for param in model.parameters():
#          print(param.requires_grad)
    
#     running_loss = 0.0
#     train_accuracy=0
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs = inputs.type('torch.FloatTensor')
#         labels = labels.type('torch.FloatTensor')
        
#         # inputs = inputs.to(memory_format=torch.contiguous_format)
#         inputs = inputs.to(device)
#         labels = labels.to(device)
        

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # Print statistics
#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}, Training Loss: {running_loss / (len(Y))}")

#     # Validation
#     model.eval()
    

#     # custom hyperparameters for validation
#     lb,ub = 0.2,0.7
#     max_val_accuracy=0
#     min_loss= 34567890.0
#     hp = (0,0)
#     with torch.no_grad():
#         for theta_b in np.arange(lb,ub,0.2):
#             for theta_i in np.arange(lb,ub,0.2):
#                 val_accuracy = 0
#                 # define thresholds
#                 val_loss = 0.0
#                 correct = 0
#                 total = 0
#                 for inputs, labels in val_loader:
#                     # print(labels)
#                     inputs = inputs.type('torch.FloatTensor')
#                     labels = labels.type('torch.FloatTensor')

#                     inputs = inputs.to(device)
#                     labels = labels.to(device)
                    
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     val_loss += loss.item()

#                     total+=1

#                     # add custom evaluation
#                     # print(theta_b)

#                     for num,output in enumerate(outputs):
#                         # for none
#                         if output[0]>0.5:
#                             val_accuracy+=labels[num][0]
#                         else:
#                             if output[1]>theta_b and output[2]>theta_i:
#                                 val_accuracy+=(labels[num][1] and labels[num][2])
#                             elif output[1]>theta_b:
#                                 val_accuracy+=labels[num][1]
#                             elif output[2]>theta_i:
#                                 val_accuracy+=labels[num][2]
#                 val_accuracy /= len(vy)
#                 val_loss /= len(vy)
#                 # print(len(val_loader))
#                 # print(f"{epoch+1} VAL_ACC : {val_accuracy} , VAL_LOSS = {val_loss}")
#                 if max_val_accuracy<val_accuracy:
#                     max_val_accuracy = val_accuracy
#                     hp = theta_b,theta_i
#                 if min_loss > val_loss : 
#                     min_loss = val_loss
#                 print(f"{epoch + 1} => MAX_ACCURACY : {max_val_accuracy} | Best HP : {hp} | MIN_LOSS : {min_loss}",flush=True)
#     wandb.log({"train_loss":running_loss / (len(vy)),"train_accuracy":train_accuracy/(len(vy)),"val_loss":min_loss,"val_accuracy":max_val_accuracy})
#     print(f"{epoch + 1} : Best HP (based on val_accuracy) : {hp}",flush=True)
#     torch.save(model, f'/scratch/fine_tune+{epoch+1}_model_cnn.pth')

# wandb.finish()
