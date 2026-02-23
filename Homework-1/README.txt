This text contains a description of how this first assignment was completed. As required, it explains:
- significance of each parameter and how its value impacts the algorithm;
- situations where the algorithm may fail.

================================================================================
CV2.KMEANS() PARAMETERS EXPLANATION
================================================================================

The following is an explanation of the parameters used in the cv2.kmeans() function call:

Function call: cv2.kmeans(flat_img, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

1. `samples` (flat_img):
   - Significance: It is the input data for clustering. Here, it is an array of (A, B) pixel values from the LAB Color Space converted to `np.float32`.
   - Impact: As explained in class, to perform segmentation based on color features, it is often advantageous to convert the image representation from the standard RGB space to the CIELAB color space. In this homework, the L channel (Lightness) is discarded, and only the A (Green-Red) and B (Blue-Yellow) channels are kept. By isolating the color information from the lighting conditions, the algorithm becomes much more robust to shadows and difficult illumination conditions. The two channels are then flattened into an array of pixels and fed into a K-Means clustering algorithm.

2. `K` (5):
   - Significance: The number of clusters to split the data into.
   - Impact: Determines the granularity of the segmentation. A higher K allows for finer color separation but increases computational cost and risks over-segmenting a single object into multiple clusters. The choice of the value of K=5 is based on the observation that while some images may have few colors, a K value of 5 ensures that complex images with many colors are handled appropriately without merging distinct colors too aggressively. Of course, this will separate the red region into multiple clusters in images with few colors (e.g. image 3). To solve this problem, the code automatically identifies any cluster whose centroid has an 'A' value greater than or equal to 145 (higher A values correspond to higher redness) and groups all such clusters together to create a unified binary mask of the red regions in the image.

3. `bestLabels` (None):
   - Significance: Optional input integer array that stores the initial cluster indices for every sample. 
   - Impact: Passing `None` tells the algorithm to initialize the labels itself. Note: the guide did not focus on this parameter, and as a consequence, I did not deeply investigate it.

4. `criteria` ((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)):
   - Significance: The algorithm termination criteria. It consists of the type, max iterations (10), and desired accuracy/epsilon (1.0).
   - Impact: The algorithm will stop iterating either when 10 iterations are reached OR when the cluster centers move by less than 1.0 unit. This parameter directly controls the trade-off between execution speed and convergence precision. Note: this was the criteria used in the guide, and it proved to work well for this specific application.

5. `attempts` (10):
   - Significance: The number of times the algorithm is executed using different initial cluster assignments. The algorithm returns the best configuration (lowest compactness).
   - Impact: Since standard K-means can get stuck in local minima, running it 10 times increases the probability of finding the optimal, true global color clusters, at the cost of processing time. Again, this was the value used in the guide, and it proved to work well for this specific application.

6. `flags` (cv2.KMEANS_PP_CENTERS):
   - Significance: The flag indicating how initial centers are chosen.
   - Impact: `cv2.KMEANS_PP_CENTERS` uses the K-Means++ initialization algorithm. It seemed to be more effective than the standard random initialization: `cv2.KMEANS_RANDOM_CENTERS`. 

================================================================================
SITUATIONS WHERE THIS K-MEANS ALGORITHM MAY FALTER
================================================================================

The algorithm successfully isolated the stop sign in all the 24 images, however, during the implementation of the project several difficulties were encountered.

1. Difficult Lighting Conditions: Several images (e.g. images 9, 16, 18) are characterized by a high level of darkness or backlight, making it challenging to isolate the stop sign. This was addressed by converting the image into LAB color space, which is more resilient to lighting changes.

2. Overly Complex Scenes (K=5 is too small): The first images are very simple and characterized by a low number of colors: for these images a K value of 3 worked well. Other images (e.g. images 16, 17, 23) contain a massive variety of colors and K=3 was not enough to separate the red regions from the background. Therefore K was increased to 5. As a consequence, this algorithm probably will not work with other images with a higher resolution and a higher number of pixels.

3. Camouflage / Background Clutter: When the target sign is located in front or near a background with a very similar shade of red (images 4, 9, 16 and 17), K-means grouped them into the same cluster. Therefore the resulting contour combined different elements: in image 17 for instance both the STOP sign and the ALL WAY sign. This was addressed by increasing the K value and filtering out the contour with an aspect ratio different from the one of a stop sign.

4. Lack of Spatial Awareness: K-Means clustering segments pixels entirely based on their color values and ignores their physical location. For instance, in images 4 and 9, the red pixels of the stop sign were clustered together with all the other red elements: the other red signs in image 4 and the traffic lights in image 9. As a consequence, the algorithm cannot separate objects by location alone and I relied on geometric filtering (area and aspect ratio) to isolate the actual sign.