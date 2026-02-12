#%%
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt


def get_box(img):
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4 
    compactness, labels, centers = cv2.kmeans(
        data=pixel_values, 
        K=k, 
        bestLabels=None, 
        criteria=criteria, 
        attempts=10, 
        flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    best_cluster_index = 0
    max_redness = -99999

    for i in range(len(centers)):
        b, g, r = centers[i]
        redness_score = int(r) - ((int(b) + int(g)) / 2)
        
        if redness_score > max_redness:
            max_redness = redness_score
            best_cluster_index = i

    labels_reshaped = labels.flatten().reshape((img.shape[0], img.shape[1]))
    mask = (labels_reshaped == best_cluster_index).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = (0, 0, 0, 0)
    max_area = 0

    if contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignoriamo rumore troppo piccolo
            if area < 500: 
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            
            # Un segnale di STOP è grosso modo quadrato.
            # Aspect ratio = w / h. Deve essere vicino a 1 (es. tra 0.5 e 1.5)
            aspect_ratio = float(w) / h
            
            # Nella tua foto errore, il box blu era l'orizzonte (molto largo, basso).
            # Quello avrebbe un aspect ratio tipo 3.0 o 4.0 -> Verrebbe scartato qui.
            if 0.5 < aspect_ratio < 1.5:
                if area > max_area:
                    max_area = area
                    best_box = (x, y, x + w, y + h)

    return best_box

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
# %%
