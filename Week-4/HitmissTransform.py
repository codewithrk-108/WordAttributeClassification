import cv2
import numpy as np
import matplotlib.pyplot as plt

fig,axs = plt.subplots(1,2,figsize=(10,20))

input_image = np.array((
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 255, 255, 255, 0, 0, 0, 255],
 [0, 255, 255, 255, 0, 0, 0, 0],
 [0, 255, 255, 255, 0, 255, 0, 0],
 [0, 0, 255, 0, 0, 0, 0, 0],
 [0, 0, 255, 0, 0, 255, 255, 0],
 [0,255, 0, 255, 0, 0, 255, 0],
 [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

kernel = np.array((
    [0,1,0],
    [1,-1,1],
    [0,1,0],
),dtype="int")

output_image = cv2.morphologyEx(input_image,cv2.MORPH_HITMISS,kernel)
print(axs[0],axs[1])

axs[0].imshow(input_image,cmap='gray')
axs[0].axis('off')
axs[0].set_title('Orignal Image')

axs[1].imshow(output_image,cmap='gray')
axs[1].axis('off')
axs[1].set_title('HitorMiss Transform')

plt.show()