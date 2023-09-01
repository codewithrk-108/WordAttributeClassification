import numpy as np
from PIL import Image
import os
from multiprocessing import Pool, cpu_count
import cv2

def resize_aspect(orignal_image):
    w = orignal_image.shape[1]//3
    h = orignal_image.shape[0]//3
    # print(w,h)
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

def aspect_conservative_resize(orignal_image,width=1200):
    w = int(width)
    h = int(orignal_image.shape[0]*(width/orignal_image.shape[1]))
    if h>170:
        w=int(170*orignal_image.shape[1]/orignal_image.shape[0])
        h = 170
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

def aspect_conservative_resize_height(orignal_image,height=170):
    h = int(height)
    w = int(orignal_image.shape[1]*(height/orignal_image.shape[0]))
    if w>1200:
        h=int(1200*orignal_image.shape[0]/orignal_image.shape[1])
        w = 1200
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

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


# Create or load your numpy arrays (for demonstration purposes)
# arrays = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
path = './test'
arrays = []

for fold in os.listdir(path):
    pth = os.path.join(path,fold)
    for img in os.listdir(pth):
        arrays.append(os.path.join(pth,img))


print(len(arrays))
# Create a destination directory if it doesn't exist
dest_dir = os.path.join(".", "new_test")
# if not os.path.exists(dest_dir):
#     os.makedirs(dest_dir)

ct=0
def save_array_as_image(arr_string):
    global ct
    split_list = arr_string.split('/')
    fold = split_list[-2]
    arr = Image.open(arr_string)
    arr = np.array(arr)
    # print(arr.shape)
    # perform ops here
    ct+=1
    
    img = centralizer(arr)
    img = resize_aspect(img)
    img_gray = Image.new('L',(img.shape[1],img.shape[0]))
    img_gray.putdata(img.flatten())
    img = img_gray
    # # print(img.shape)
    print(f"\r{ct}",end='',flush=True)
    # # print(img.shape)
    des = os.path.join(dest_dir,f"{fold}")
    image_path = os.path.join(des, f"{split_list[-1]}")
    # img = Image.fromarray(img)
    img.save(image_path)
    return f"cool"
    # return f"Saved: {image_path}"

if __name__ == "__main__":
    # print(cpu_count())
    # with Pool(processes=cpu_count()) as pool:
    results = list(map(save_array_as_image, arrays))
    print(len(results))
