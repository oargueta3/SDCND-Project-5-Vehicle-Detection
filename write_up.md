## **SDCND Project 5: Vehicle Detection and Tracking**

### Objective
An important function of an autonomous vehicle is to be able to detect and track vehicles around it. For this project, the objective is to explore a method of detecting vehicles using a linear support vector machine (SVM) and write a software pipeline to detect vehicles on a continous video stream.
### Overview
The following are the steps taken to build a vehicle detection pipeline::

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
2. Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
3. Normalize features and randomize a selection for training and testing.
4. Train SVM Classifier.
5. Implement a sliding-window technique and use the trained classifier to search for vehicles in image frames
6. Run the frame processing pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
7. Estimate and draw a bounding box for vehicles detected on the video stream.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hogs.png
[image3]: ./output_images/raw.png
[image4]: ./output_images/color_bin.png
[image5]: ./output_images/car_hist.png
[image6]: ./output_images/not_car_hist.png
[image7]: ./output_images/slide_window.png
[image8]: ./output_images/search_detect.png
[image9]: ./output_images/heatmap.png
[image10]: ./output_images/algo_diagram.png
[image12]: ./output_images/f_vec.png



### Feature Extraction: Spatial Binning of Color, Color Historgrams, and Histogram of Oriented Gradients (HOG)
---
A crucial step to having a high performance classifier is selecting the right features for training that result in high accuracy and speed. For each image in the labeled dataset a feature vector was constructed to train an SVM classifier. The selected features include:

1. **Raw Pixel Values with Spatial Binning**: Raw pixel data of a reduced resolution image. Even with reduced resolution key structure information is kept.
2. **Color Histogram**: Extract information for color distribution (shape invariance).
3. **HOG - Histogram of Gradients**: Extract information related to shape structure.

After all the features are extracted, they are concatanated together to form a single feature vector. Eac h image has its own vector and these are used to train classifiers. Below is a breakdown of the contribution of each feature to the final feature vector for each image sample.

* **Feature from spatial binning**: 3017
* **Features from Color Histogram** : 96
* **Features from HOG from 3 channels**: 5292
* **Total # of Features**: 8460

It is important to note that training data is divided into two classes, vehicle and non-vehicle images. Here is an example of one of each:

![car not car][image1]

These images can be found in the `vehicle` and `non-vehicle` directories

#### Feature 1: Spatially Binning of Color
The first component of the feature vector of each image in the labeled dataset is the raw pixel information. An important observation with objects such as cars is that they have high color saturation. It is possible to maximize the information extracted from these pixels by selecting the appropriate color space. With the correct selection, it is possible to identify clusters of pixels that can help us differentiate cars from non-car objects. After testing several color spaces (RGB, HLS, HSV, YUV, etc), **YCbCr** showed clearest clusters with the least of amount of noise. 3D visualization of pixel values in YCbCr Space can be seen below for vehicle and non-vehicle images:

![YCbCr in 3D Space][image3]

While it could be computationally intensive to include three color channels of a full resolution image, you can perform **spatial binning** on an image and still retain enough information to help in finding vehicles.
A convenient function for scaling down the resolution of an image The OpenCV function `cv2.resize()` scales down the resolution of the image. To extract this feature images were resized from 64x64 to 32x32. An examples of a spatially binned image can be seen below.

![bin car][image4]

The code below was used to obtain the spatially binned feature vector of images:
```
def spatial_bin(image, new_size=(32,32)):
    features = cv2.resize(image, new_size).ravel()
    return features
```
This code can also be found in code cells 9 and 10 in the `Vehicle Detection and Tracking - Oscar Argueta.ipynb` **IPython notebook**.

#### Feature 2: Color Histogram

The distribution of pixels/color across channels can provide shape invariant information about objects in an image. By picking the right color space useful information extraction can be maximized. The color histogram from vehicle and non-vehicle images was computed accross several color spaces. Color Spaces that provide information about color saturation seemed to provide the best signatures overall; therefore **YCbCr** was selected to keep the feature extraction pipeline simple. Below, an example of the histogram of each color channel for vehicle and non-vehicle images:

###### **Vehicle Color Histogram YCbCr Color Space**

![car hist][image5]

###### **Non-Vehicle Color Histogram YCbCr Color Space**

![ not car hist][image6]

The code below was used to obtain the color histogram feature vector of images:

```
def color_hist(img, nbins=32, bins_range=(0, 256), hist_channels='ALL', vis=False):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    if vis == True:
        # Generate bin centers for plotting
        bin_edges = channel1_hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features
    else:
        # Return color histogram features of the specified amount of channels
        if hist_channels == 1:
            hist_features = channel1_hist[0] 
        elif hist_channels == 2:
            hist_features = np.concatenate((channel1_hist[0], channel2_hist[0]))
        elif hist_channels == "ALL":
            hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0])) 
        return hist_features
```

Code realated to color histograms can also be found in code cells 12 through 15 in the `Vehicle Detection - Oscar Argueta.ipynb` **IPython

#### Feature 3: Histogtam of Oriented Gradients (HOG)
Histograms of Oriented Gradients provide key shape information about object within images. Using the Python package `skimage` built-in method `hog()`, random images from the vehicle and non-vehicle datasets were processed using different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) and color spaces. In order to pick the optimal parameters a SVM classifier was trained only with HOG features. The parameters that yielded the greatest performance on the test set were selected. Below are the selected parameters: 

**Selected Parameters**
* `orientation_bins`: 9
* `pixels_per_cell`: (8 x 8)
* `cells_per_block`: (2 x 2)
* `hog_channels`: 3
* `color_space`: YCbCr

Here is an example of HOG features using the `YCrCb` color space and the selected HOG parameters for vehicle and non-vehicle images:

![hog][image2]

The code below was used to to extract the HOG feature vector from images:
```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

Code realated to HOG can also be found in code cells 16 through 20 in the `Vehicle Detection and Tracking - Oscar Argueta.ipynb` **IPython notebook**.


### Support Vector Machine(SVM) Classifier
---
A support vector machine(SVM) was used to classify the contents of image frames to locate vehicle objects in a video stream. To train clasiffier, a dataset containing examples of vehicle and non-vehicle images was used. From each sample three, features (HOG, Color Histogram, Spatial Binning) were extracted into a single 1-dimensional feature vector as seen below.

![ not car hist][image12]

The code below was used to extract a similar feature vector from each image:
```
def extract_image_features(image_file, cspace='RGB', spatial_size=(32,32),hist_bins=32, hist_range=(0,256), 
                     hist_channels=2, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """pydoc"""
    # Apply color conversion if other than 'RGB'
    image = read_image(image_file)
    if cspace != 'RGB':
        cv2_space = eval("cv2.COLOR_RGB2" + cspace)
        feature_image = cv2.cvtColor(image, cv2_space)
    else: feature_image = np.copy(image)      

    # Apply spatial_bin() to get spatial color features
    spatial_features = spatial_bin(feature_image, new_size=spatial_size)
        
    # Apply color_hist() to get histogram features
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range, hist_channels=hist_channels)
    
    # Apply get_hog_features() to obtain HOG features
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    
    # Concatanate all features into a single vector
    features = np.concatenate((spatial_features, hist_features, hog_features))
    return features
```

After extracting all features from the vehicle and non-vehicle datasets, the following steps were taken
to prepare the training data for the SVM:

1. Combine vehicle and non-vehicle data
2. Normalize feature vectors using the `StandardScaler()` from `sklearn`
3. Create  matching training labels for each feature vector (1 for car, 0 for no car)
4. Shuffle and create a train-test split

The code below shos the steps discussed above:

```
# Stack features
X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)

# Normalize features - Scale Features
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# Create labels
Y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
               
# Shuffle and train-test split
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.2, random_state=rand_state)
```

After training with default parameters, the SVM achieved an accuracy of **0.9977** (99.77%) on the test set.

Code realated to the SVM can also be found in code cells 21 through 28 in the `Vehicle Detection and Tracking - Oscar Argueta.ipynb` **IPython notebook**.


### Sliding Window Search
---

Slide window search takes a patch of an image (**The patch is the region of the image where vehicles are expected
to be seen**) and divides it into overlapping fixed sized windows. The coordinates of each window/box are 
recorded and its contents will be put through an SVM to check whether a vehicle is present. Below is a graphical
represenation of slide window search over a predetermined region of interest.

![slide][image7]

The challenge with slide window search is that vehicles appear to be of different sizes depending on their distance 
from the camera. In order to capture varying sizes of vehicles without having to use different sizes of windows,
the `xy_window` and the `xy_overlap` parameters were tuned. By selecting a relatively large window with high
overlap percentage, the SVM is able to detect vehicles of different sizes robustly. After testing several
`xy_window` and the `xy_overlap` the following values were selected based on a healthy number of overlapping
windows detecting the same vehicle. 

The parameters choses are the following:

1. `x_start_stop` = `[None, None]`
2. `y_start_stop` = `[380, 625]`
3. `xy_window` = `(80, 80)` 
4. `xy_overlap` = `(0.75, 0.75)`

Code realated to the Slide Window Search can also be found in code cells 30 and 31 in the `Vehicle Detection and Tracking - Oscar Argueta.ipynb` **IPython notebook**.


### Search and Classify
---

Using the coordinates of the windows/boxes obtained from Slide Window Search, the features of each window were
extracted and ran through a SVM classifier to determine whether a vehicle is present in it. Windows with a vehicle
detected in them are called hot windows. The coordinates of all hot windows are recorded to build a heat map of
detectionts. It is important to note that windows of only one size were used to detect vehicles of different sizes.
This was possible by selecting appropriate `xy_window` and the `xy_overlap` parameters in the Slide Window Search.
Below are examples of overlapping hot windows over vehicles:


![detect][image8]


The code below was used to detect hot windows:

```
def search_for_cars(image, window_boxes, svc, scaler, cspace='RGB', spatial_size=(32, 32), hist_bins=32, 
hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):

# Extract all window boxes where a vehicle was detected
hot_windows = []
for box in window_boxes: 

# Extract image patch from original image and resize to 64 x 64
# Images used to train the model were all of size 64 by 64.
# It is imperative to extract the same amount of features so that the classifier works correctly
image_patch = cv2.resize(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], (64, 64))      

# Extract features from current image in frame
features = extract_image_features(image_patch, cspace, spatial_size, hist_bins, hist_range, 
hist_channels, orient, pix_per_cell, cell_per_block, hog_channel)

# Scale extracted features to be fed to the classifier
test_features = scaler.transform(np.array(features).reshape(1, -1))

#Predict using your classifier
prediction = svc.predict(test_features)
# If positive (prediction == 1) then save the window
if prediction == 1:
hot_windows.append(box)

return hot_windows
```

Code realated to search and classfication  can also be found in code cells 32 through 34 in the `Vehicle Detection and Tracking - Oscar Argueta.ipynb` **IPython notebook**.

### Heat Maps and Outlier Removal
--- 

In order to remove outliers(false positives) from random frames, the hot windows were used to build a heatmap
such that areas of multiple detections get "hot", while areas with false positives stay "cool" and effectively
remove the outlier

The steps to build a heatmap are the following:
1. Create a matrix of zeros of the size of the image
2. Add 1 to each pixel location contained in each hot window
3. Threshold the heatmap to remove "cool" values

The code below was used for building and thresholding heat maps:
```
# Function to populate Heatmap
def add_heat(heatmap, bbox_list):
# Iterate through list of bboxes
for box in bbox_list:
# Add += 1 for all pixels inside each bbox
# Assuming each "box" takes the form ((x1, y1), (x2, y2))
heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

# Return updated heatmap
return heatmap

# Function to apply threshold to the heat map to discard cool zones - areas where vehicles are not present
def apply_threshold(heatmap, threshold):
# Zero out pixels below the threshold
heatmap[heatmap <= threshold] = 0
# Return thresholded map
return heatmap

```

The next step was to use the `scipy.ndimage.measurements.label()` function to to identify individual blobs in the
heat map. Under the assumption that each blob corresponded to a vehicle, bounding boxes were constructed to cover the
area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video and the bounding boxes overlaid on
original image frame:
 
![heat][image9]

Code realated to heat maps can also be found in code cells 35 through 37 in the `Vehicle Detection and Tracking - Oscar Argueta.ipynb` **IPython notebook**.

### Video Implementation
---
The function **`process_frame()`** returns the original image frame with bounding boxes drawn around the vehicles detected. A special class called `HeatQueue` was made to track and sum the heatmaps of 25 frames to optimize outlier removal and smoother temporal performance. A flow diagram describing the function **`process_frame()`** is seen below.

![algo][image10]

The code for the `HeatQueue` class can be seen below:


Code realated to the `process_frame()` function  can also be found in code cells 38 through 42 in the `Vehicle Detection and Tracking - Oscar Argueta.ipynb` **IPython notebook**.


#### Final Output (video) 

<a href="https://www.youtube.com/embed/ztvgLcQjkjc target="_blank"><img src="http://img.youtube.com/vi/ztvgLcQjkjc/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>


---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


- notes for conclusion
Very innefficient, would have to search and extract HOG features for each window. HOG is an expensive operation and since the "Search boxes" overlap we would be doing 2X unnecessary computations.
The total number of windows to search per frame: **610*

