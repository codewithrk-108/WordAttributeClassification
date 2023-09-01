import torch
import torchvision.models as models
from torchvision.transforms import ToTensor
import torch.nn as nn
import h5py
import gc
import numpy as np
import json
import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
import os

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(75 * 10 * 128, 256)
        # dropout1 = nn.Dropout(p=0.5) 
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # print(x.shape)
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        # print(x.shape)
        
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        # print(x.shape)
        
        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))
        # print(x.shape)
        
        x = self.flatten(x)
        # print(x.shape)
        
        
        x = self.bn4(torch.relu(self.fc1(x)))
        # print(x.shape)
        
        x = self.bn5(torch.relu(self.fc2(x)))
        # print(x.shape)
        
        x = self.fc3(x)
        # print(x.shape)
        
        x = self.sigmoid(x)
        return x


with open('/scratch/bbox_info.json','r') as file:
	bbox_info = json.load(file)

with open('/scratch/rev_map.json','r') as file:
	rev_map = json.load(file)	

filename = '/scratch/bbox_pixels.hdf5'
with h5py.File(filename, 'r') as hf:
		# print(synthetic_filename[0:-5])
	name = filename.split('/')[-1].split('.')[-2]
	inputs = hf[name][:]

print(inputs[1])
inputs = inputs/255
inputs = np.array(inputs)
print(inputs.shape)

# model = torch.load('/scratch/.h5')
# pred = model.predict(inputs)
# print(pred.shape)
# print(np.unique(pred))


# counter={0:0,1:0,2:0,3:0}

# sns.kdeplot(np.clip(pred[:,0],0,1))
# plt.savefig("none.png")
# sns.kdeplot(np.clip(pred[:,1],0,1))
# plt.savefig("bold.png")
# sns.kdeplot(np.clip(pred[:,2],0,1))
# plt.savefig("italic.png")
# sns.kdeplot(np.clip(pred[:,3],0,1))
# plt.savefig("semibold.png")

# print(np.median(pred[:,0]),np.max(pred[:,0]))
# print(np.median(pred[:,1]),np.max(pred[:,1]))
# print(np.median(pred[:,2]),np.max(pred[:,2]))
# print(np.median(pred[:,3]),np.max(pred[:,3]))
# # pred[:,0] = (pred[:,0]-np.min(pred[:,0]))/(np.max(pred[:,0])-np.min(pred[:,0]))
# # pred[:,1] = (pred[:,1]-np.min(pred[:,1]))/(np.max(pred[:,1])-np.min(pred[:,1]))
# # pred[:,2] = (pred[:,2]-np.min(pred[:,2]))/(np.max(pred[:,2])-np.min(pred[:,2]))
# # pred[:,3] = (pred[:,3]-np.min(pred[:,3]))/(np.max(pred[:,3])-np.min(pred[:,3]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model = model.to(device)

transform = ToTensor()

model = torch.load('/scratch/5_model_cnn.pth')
model.to(device)

out_final = []

with torch.no_grad():
	for i in range(0,inputs.shape[0],50):
                arr = inputs[i:i+50]
                tens=[]
                for i in arr:
                                tens.append(transform(i))
                tens = np.array(tens)
                print(tens.shape)
                tens = torch.from_numpy(tens)
                tens = tens.to(device)
                print(tens.device)
                outputs = model(tens)
                for i in outputs:
                        out_final.append(i)
                print("done")

for cnt,pre in enumerate(out_final):
                for inde_dict in (bbox_info[rev_map[str(cnt)]['file']][rev_map[str(cnt)]['index']]['bb_ids']):
                         if inde_dict["id"]==cnt:
                              if pre[0]>=0.5:
                                   break
                              if pre[1]>0.4 and pre[2]>0.4:
                                   inde_dict["attb"]["isBold"]=True
                                   inde_dict["attb"]["isItalic"]=True
                              elif pre[1]>0.4:
                                   inde_dict["attb"]["isBold"]=True
                              elif pre[2]>0.4:
                                   inde_dict["attb"]["isItalic"]=True

with open('/scratch/bbox_info_res.json','w') as file:
	json.dump(bbox_info,file)
