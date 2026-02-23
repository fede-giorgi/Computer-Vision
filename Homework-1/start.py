#%%

import cv2
import os
import time

# ADDITIONAL IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def debug_plot_kmeans(img_lab, center, label, K):
    """
    Helper function used during development to visualize the results of the K-means clustering.
    It reconstructs the image using the cluster centers and plots it with a custom legend.
    NOTE: This function is kept to show the thought process but is NOT called during execution to save time.
    """
    # Select only the A and B channels from the LAB image for shape reference
    ab_channel = img_lab[:, :, 1:3]
    
    # Map each pixel label back to its corresponding cluster center color
    res = center[label.flatten()]
    
    # Reshape the flattened result back to the 2D image dimensions
    res2 = res.reshape((ab_channel.shape))
    
    # Extract the clustered A and B channels
    a_clustered = res2[:, :, 0]
    b_clustered = res2[:, :, 1]
    
    # Create a dummy L channel filled with a constant value (170) for visualization purposes
    l_channel = np.full(a_clustered.shape, 170, dtype=np.uint8)

    # Merge the channels back into a LAB image
    lab_reconstructed = cv2.merge((l_channel, a_clustered, b_clustered))
    
    # Convert the reconstructed LAB image to RGB for plotting with matplotlib
    rbg_reconstructed = cv2.cvtColor(lab_reconstructed, cv2.COLOR_LAB2RGB)

    # Plot the reconstructed image
    plt.imshow(rbg_reconstructed)
    plt.title('K-Means results (Reconstructed)')

    # Initialize a list to hold the legend patches
    legend_patches = []

    # Loop through each cluster to create a colored patch for the legend
    for k in range(K):
        # Extract the A and B values for this specific cluster center
        a_val = center[k, 0]
        b_val = center[k, 1]
        
        # Create a dummy 1x1 pixel with the LAB color of this cluster
        cluster_lab_pixel = np.uint8([[[170, a_val, b_val]]])
        
        # Convert the pixel to RGB for Matplotlib compatibility
        cluster_rgb_pixel = cv2.cvtColor(cluster_lab_pixel, cv2.COLOR_LAB2RGB)[0][0]
        
        # Normalize the color values to the [0, 1] range required by Matplotlib
        color_normalized = cluster_rgb_pixel / 255.0
        
        # Create the text label showing the cluster index and its A channel value
        label_text = f"Cluster {k}: A={a_val}" 
        
        # Create the "square" (patch) for the legend with the cluster color
        patch = mpatches.Patch(color=color_normalized, label=label_text)
        legend_patches.append(patch)

    # Add the legend to the plot and display it
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() 
    plt.show()


def debug_plot_contours(img_rgb, contours):
    """
    Helper function used during development to visualize all detected contours on the original image.
    NOTE: This function is kept to show the thought process but is NOT called during execution.
    """
    # Create a copy of the original RGB image to draw contours on
    img_contours = img_rgb.copy()
    
    # Draw all detected contours in green with a thickness of 3
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # Plot the image with contours
    plt.imshow(img_contours)
    plt.title("All Contours")
    plt.show()


def get_box(img):
    """
    Analyzes the input BGR image, isolates the red color using K-Means clustering in the LAB color space,
    finds the contour of the stop sign based on area and aspect ratio, and returns its bounding box.
    """
    # Apply median filter with a kernel size of 7 to reduce noise while keeping edges sharp
    img_medianfiltered = cv2.medianBlur(img, ksize=7)

    # Convert the BGR image to the LAB color space to separate color information from brightness
    img_lab = cv2.cvtColor(img_medianfiltered, cv2.COLOR_BGR2LAB)

    # Select only the A (Green-Red) and B (Blue-Yellow) channels, discarding Lightness (L)
    ab_channel = img_lab[:, :, 1:3]

    # Flatten the 2D image into a 1D array of pixel pairs (A, B) for clustering
    flat_img = ab_channel.reshape((-1, 2))
    
    # Convert the flattened array to float32 type, which is required by the cv2.kmeans function
    flat_img = np.float32(flat_img)

    # Define the criteria for the K-means algorithm: stop after 10 iterations OR if centers move by less than 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Set the number of clusters
    K = 5

    # Perform K-means clustering to group similar colors together
    ret, label, center = cv2.kmeans(
        flat_img, 
        K, 
        None, 
        criteria, 
        20, # Number of attempts with different initial cluster centerings
        cv2.KMEANS_PP_CENTERS # Use K-means++ initialization for better center selection
    )

    # Convert the resulting cluster centers back to unsigned 8-bit integers
    center = np.uint8(center)

    # Find the indices of clusters where the A channel value (Red/Green axis) is >= 145, indicating a strong red color
    red_cluster_indices = np.where(center[:, 0] >= 145)[0]
   
    # Create an empty 1D array (mask) of zeros with the same size as the flattened labels
    mask_flat = np.zeros_like(label.flatten(), dtype=np.uint8)
    
    # Loop through each index that was identified as a "red" cluster
    for idx in red_cluster_indices:
        # Turn on (set to 255) the pixels in the mask that belong to this specific red cluster
        mask_flat[label.flatten() == idx] = 255

    # Reshape the flattened 1D mask back to the original 2D image dimensions
    mask = mask_flat.reshape((img_lab.shape[0], img_lab.shape[1]))

    # Find the contours of the red regions using the binary mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a variable to store the contour that best fits the stop sign description
    best_contour = None
    
    # Initialize a variable to keep track of the largest contour area found so far
    max_area = 0

    # Loop through all the contours detected in the mask
    for cnt in contours:
        # Calculate the area of the current contour
        area = cv2.contourArea(cnt)
        
        # Ignore very small contours to filter out remaining noise
        if area > 100: 
            # Get the bounding rectangle coordinates (x, y) and dimensions (width, height) for the contour
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate the aspect ratio (width divided by height) of the bounding box
            aspect_ratio = float(w) / h
            
            # Check if the aspect ratio is close to 1 (between 0.64 and 1.36). Stop signs are roughly square in bounding boxes.
            if 0.64 <= aspect_ratio <= 1.36:
                    
                # If this is the largest valid contour found so far, update max_area and set it as the best_contour
                if area > max_area:
                    max_area = area
                    best_contour = cnt

    # If a valid contour was successfully found
    if best_contour is not None:
        # Get the bounding rectangle coordinates and dimensions for the best contour
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Calculate xmin, ymin, xmax, and ymax needed for drawing the final rectangle
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        
        # Return the four coordinates representing the bounding box
        return xmin, ymin, xmax, ymax

    # If no valid contour was found, return zeros to avoid program crashes
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