The following text contains a description and explanation of the approach and 
strategy used to complete the first assignment.


================================================================================
APPROACH EXPLANATION
================================================================================
The approach can be broken down into the following steps:

1. Median Filter: A median filter is applied to the image first. This is highly 
    effective at removing noise without blurring the edges of the underlying 
    objects. The kernel size is set to 7, this choice is based on a elboow method 
    analysis of the trade-off between noise reduction and edge preservation: 
    several values were tested and 7 was found to be the most effective.

2. LAB Color Space: The image is converted from BGR to the LAB color space. 
    The L channel (Lightness) is discarded, and only the A (Green-Red) and B 
    (Blue-Yellow) channels are kept. By isolating the color information from 
    the lighting conditions, the algorithm becomes much more robust to shadows 
    and illumination changes. The two channels are then flattened into an array 
    of pixels and fed into a K-Means clustering algorithm.

3. K-Means Clustering and Red Color Extraction: The function call is cv2.kmeans(flat_img, K, None, criteria,
    20, cv2.KMEANS_PP_CENTERS) where K is set to 5. This choice is based on the observation
    that while some images may have few colors, a K value of 5 ensures that complex 
    images with many colors are handled appropriately without merging distinct colors 
    too aggressively. Of course, this will separate the red region into multiple clusters 
    in images with few colours (e.g. image 3). The code automatically identifies any cluster 
    whose centroid has an 'A' value greater than or equal to 145 (higher A values correspond 
    to higher redness) and groups all such clusters together to create a unified binary mask 
    of the red regions in the image.

4. Contour Detection: The code finds the boundaries (contours) of the red regions identified in 
     the binary mask. Those contours are then filtered based on two geometric properties:
     a) Area: Must be greater than 100 pixels (to ignore small noise blobs).
     b) Aspect Ratio (Width / Height): Must fall between 0.64 and 1.36, in order 
     to isolate objects that are roughly square or circular (as a stop sign).
     Finally, out of all contours that pass these filters, the one with the 
     maximum area is selected as the best candidate, and a bounding box is 
     drawn around it.


================================================================================
CV2.KMEANS() PARAMETERS EXPLANATION
================================================================================
Function call: 
cv2.kmeans(flat_img, K, None, criteria, 20, cv2.KMEANS_PP_CENTERS)

1. `data` (flat_img):
   - Significance: The input data for clustering. Here, it is an array of 
     (A, B) pixel values converted to `np.float32`.
   - Impact: This defines the feature space. By omitting the 'L' channel, the 
     clustering is strictly based on chromaticity making easier to isolate 
     the red regions in images with different lighting conditions (like the
     sunset in image 16).

2. `K` (5):
   - Significance: The number of clusters to split the data into.
   - Impact: Determines the granularity of the segmentation. A higher K allows 
     for finer color separation but increases computational cost and risks over-
     segmenting a single object into multiple clusters. As previously noted, 
     in some images with few colors, this will separate the red region into 
     multiple clusters.

3. `bestLabels` (None):
   - Significance: Optional input integer array that stores the initial cluster 
     indices for every sample. 
   - Impact: Passing `None` tells the algorithm to initialize the labels itself.

4. `criteria` ((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)):
   - Significance: The algorithm termination criteria. It consists of the type, 
     max iterations (10), and desired accuracy/epsilon (1.0).
   - Impact: The algorithm will stop iterating either when 10 iterations are 
     reached OR when the cluster centers move by less than 1.0 unit. This 
     parameter directly controls the trade-off between execution speed and 
     convergence precision.

5. `attempts` (20):
   - Significance: The number of times the algorithm is executed using different 
     initial cluster assignments. The algorithm returns the best configuration 
     (lowest compactness).
   - Impact: Since standard K-means can get stuck in local minima, running it 
     20 times significantly increases the probability of finding the optimal, 
     true global color clusters, at the cost of processing time.

6. `flags` (cv2.KMEANS_PP_CENTERS):
   - Significance: The flag indicating how initial centers are chosen.
   - Impact: `KMEANS_PP_CENTERS` uses the K-Means++ initialization algorithm. 
     This method smartly spreads out the initial centroids, which makes the 
     algorithm converge much faster and yields significantly better final clusters 
     compared to purely random initialization.


================================================================================
SITUATIONS WHERE THIS K-MEANS ALGORITHM MAY FALTER
================================================================================

The algorithm successfully isolated the stop sign in all the 24 images, however, 
during the implementation of the project several difficulties were encountered.

1. Highly Washed-out or Dark Images: Several images (e.g. images 9, 16, 18) are
characterized by a high level of darkness or difficult lighting conditions, making 
it challenging to isolate the stop sign. This was addressed by converting the image
into LAB color space, which is more resilient to lighting changes.

2. Overly Complex Scenes (K=5 is too small):
   If an image contains a massive variety of vibrant colors (e.g., a bustling 
   cityscape with neon lights), K=5 might force the algorithm to group the red 
   target object with a slightly orange or purple background object. This merged 
   cluster could end up with an average 'A' value below 150, or create a contour 
   that fails the geometric aspect ratio test.

3. Camouflage / Background Clutter: If the target object is located in front of 
   a background with a very similar shade of red (e.g., a stop sign in front of a 
   red brick building), K-means will group them into the same cluster. The 
   resulting contour will combine the sign and the building, completely ruining 
   the aspect ratio (0.64 - 1.36) filtering step.

4. Performance Overhead on High-Resolution Images: K-means clustering on a flattened 
   image array is computationally heavy. With `attempts=20` and `K=5`, applying this 
   directly to a 4K or 1080p image without downsampling first could cause the 
   algorithm to run very slowly, making it unsuitable for real-time video processing.