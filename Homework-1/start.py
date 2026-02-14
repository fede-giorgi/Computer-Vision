import cv2
import os
import time

def get_box(img):
    # Your code start here #
    # Have fun!            #
    return 0, 0, 0, 0

if __name__ == "__main__":

    start_time = time.time()

    dir_path = './images/'
    for i in range(1, 25):
        img_name = f'stop{i}.png'
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        # Get the coordinators of the box
        xmin, ymin, xmax, ymax = get_box(img)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        output_path = f'./results/{img_name}'
        cv2.imwrite(output_path, img)

    end_time = time.time()
    #It usually takes about 10s
    print(f"Running time: {end_time - start_time} seconds")

#%%

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

dir_path = './images/'
i = 2
img_name = f'stop{i}.png'
img_path = os.path.join(dir_path, img_name)
img = cv2.imread(img_path)

# convert and show RGB image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# apply median filter to remove noise
img_medianfiltered = cv2.medianBlur(img_rgb, ksize=5)

# Soluzione: 
img = cv2.cvtColor(img_medianfiltered, cv2.COLOR_RGB2LAB)

# flatten the image
flat_img = img.reshape((-1, 3))
flat_img = np.float32(flat_img)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv2.kmeans(
    flat_img,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_PP_CENTERS
)

center = np.uint8(center)
res = center[label.flatten()]
res2_lab = res.reshape((img.shape))
res2_bgr = cv2.cvtColor(res2_lab, cv2.COLOR_LAB2BGR)

plt.imshow(res2_bgr)
plt.title("K-Means Clustering")
plt.show()

# select the cluster with the maximum a* value (red color)
cluster_id = np.argmax(center[:, 1])

# create a binary mask for the red cluster
mask_flat = np.where(label.flatten() == cluster_id, 255, 0).astype(np.uint8)

# reshape the mask to the original image dimensions
mask = mask_flat.reshape((img.shape[0], img.shape[1]))

# show the binary mask
plt.imshow(mask)
plt.title("Binary Mask")
plt.show()

# find contours on the binary mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# copy the original image to draw on it
img_contours = img.copy()

# draw all the contours found (-1) in green (0, 255, 0) with thickness 3
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

# show the results
plt.imshow(img_contours)
plt.title("Contours")
plt.show()

img_boxes = img.copy()

# make sure we found at least one contour
if len(contours) > 0:
    # find the contour with the largest area using the max() function
    contorno_principale = max(contours, key=cv2.contourArea)
    
    # calculate the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(contorno_principale)
    
    # draw the bounding box
    cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 255, 0), 5)

# convert the image to RGB
img_boxes_rgb = cv2.cvtColor(img_boxes, cv2.COLOR_LAB2RGB)

# show the results
plt.imshow(img_boxes_rgb)
plt.title("Identified Sign!")
plt.show()

#%%


def show_images(img1, img2):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.title('Median Filtered Image')
    plt.imshow(img2)
    plt.show()


def show_images(img1, img2):  
    plt.imshow(img1) # crazy colors
    plt.show()
    # split LAB image into luminosity, green-red, blue-yellow channels
    L, A, B = cv2.split(img2)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('L (Luminosity) channel')
    plt.imshow(L, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('A (Green-Red) channel')
    plt.imshow(A, cmap='RdYlGn_r') 

    plt.subplot(1, 3, 3)
    plt.title('B (Blue-Yellow) channel')
    plt.imshow(B, cmap='YlGnBu_r') 

    plt.show()