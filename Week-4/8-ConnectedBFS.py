import cv2
from skiimage import data
from skimage.util import invert
import numpy as np
import matplotlib.pyplot as plt


def isBoundary(img,pos):
    x,y = pos
    # print(x,y)
    h,w = img.shape
    boundary=0

    if x-1>=0 and y-1>=0:
        boundary+=int(img[x-1][y-1]==0)
    if x+1<h and y-1>=0:
        boundary+=int(img[x+1][y-1]==0)
    if x+1<h and y+1<w:
        boundary+=int(img[x+1][y+1]==0)
    if x-1>=0 and y+1<w:
        boundary+=int(img[x-1][y+1]==0)
    if x-1>=0:
        boundary+=int(img[x-1][y]==0)
    if x+1<h:
        boundary+=int(img[x+1][y]==0)
    if y-1>=0:
        boundary+=int(img[x][y-1]==0)
    if y+1<w:
        boundary+=int(img[x][y+1]==0)
    
    return (boundary>=1)

image = invert(data.horse())
image = np.array(image,dtype='uint8')
image[image==1] = 255

# image = cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)

# print(image.shape)
# Algorithm for Distance transform

queue = []
image_pairs={}
cnt=0
for row in range(len(image)):
    for col in range(len(image[0])):
        if image[row][col]==255 and isBoundary(image,(row,col)):
            image_pairs[(row,col)]=1
            queue.append((row,col))
        else:
            image_pairs[(row,col)]=0
            # cnt+=1
# print(cnt)

h,w = image.shape

# 8 - Connected
# print(len(queue))
while len(queue)>0:
    top = queue.pop(0)
    x,y = top

    if x-1>=0 and y-1>=0 and image[x-1][y-1]==255 and image_pairs[(x-1,y-1)]==0:
        image_pairs[(x-1,y-1)]=image_pairs[top]+1
        queue.append((x-1,y-1))
    if x+1<h and y-1>=0 and image[x+1][y-1]==255 and image_pairs[(x+1,y-1)]==0:
        image_pairs[(x+1,y-1)]=image_pairs[top]+1
        queue.append((x+1,y-1))
    if x-1>=0 and y+1<w and image[x-1][y+1]==255 and image_pairs[(x-1,y+1)]==0:
        image_pairs[(x-1,y+1)]=image_pairs[top]+1
        queue.append((x-1,y+1))
    if x+1<h and y+1<w and image[x+1][y+1]==255 and image_pairs[(x+1,y+1)]==0:
        image_pairs[(x+1,y+1)]=image_pairs[top]+1
        queue.append((x+1,y+1))

    if x-1>=0 and image[x-1][y]==255 and image_pairs[(x-1,y)]==0:
        image_pairs[(x-1,y)]=image_pairs[top]+1
        queue.append((x-1,y))
    if x+1<h and image[x+1][y]==255 and image_pairs[(x+1,y)]==0:
        image_pairs[(x+1,y)]=image_pairs[top]+1
        queue.append((x+1,y))
    if y-1>=0 and image[x][y-1]==255 and image_pairs[(x,y-1)]==0:
        image_pairs[(x,y-1)]=image_pairs[top]+1
        queue.append((x,y-1))
    if y+1<w and image[x][y+1]==255 and image_pairs[(x,y+1)]==0:
        image_pairs[(x,y+1)]=image_pairs[top]+1
        queue.append((x,y+1))

# for row in range(len(image)):
    # for col in range(len(image[0])):
        # print(image_pairs[(row,col)],end=" ")
    # print()
# print(image_pairs.values())
max_val = max(image_pairs.values())

distancetr_map = np.zeros(image.shape)
for k,v in image_pairs.items():
    distancetr_map[k[0]][k[1]] = int(v*255.0/max_val)

# print(image)
# print(distancetr_map)

fig,axs = plt.subplots(1,1)
axs.imshow(distancetr_map,cmap='gray')
# cv2.imshow('Distance Map',distancetr_map)
plt.show()