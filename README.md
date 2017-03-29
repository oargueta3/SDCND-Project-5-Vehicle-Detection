# Self-Driving Car Nanodegree
# Vehicle Detection and Tracking

### Overview
An important function of an autonomous vehicle is to be able to detect and track vehicles around it. For this
project, the objective is to explore a method of detecting vehicles using a linear support vector machine (SVM) and
write a software pipeline to detect vehicles on a continous video stream. The following were the steps taken to build a vehicle detection pipeline:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
2. Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
3. Normalize features and randomize a selection for training and testing.
4. Train SVM Classifier.
5. Implement a sliding-window technique and use the trained classifier to search for vehicles in image frames
6. Run the frame processing pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
7. Estimate and draw a bounding box for vehicles detected on the video stream. 

### Included Files

This project was written in Python. The follwing files were used to create the vehicle detection pipeline:

1. `Vehicle Detection and Tracking - Oscar Argueta.ipynb`: Used to develop the vehicle detection algorithm. Detailed explanations and figures can be found in this jupyter notebook
2. `svm_pickle.p`: Pickle file containing trained SVM (Support Vector Machine) classifier and feature scaler to normalize feature vectors
3. `test_images/`: Directory containing images to test the vehicle detection pipeline at different stages
4. `output_images/`: Directory containing ouput images from the different stages of the vehicle detection pipeline
5. `write_up.md`: Detailed report that includes descriptions, figures and code used for each step of the vehicle detection pipeline
6. `test_video.mp4`: Video stream to test the vehicle detection algorithm on
6. `project_video.mp4`: Video stream to run the vehicle detection algorithm on
7. `result.mp4`: Output video with bounding boxes drawn on detected 


### Algorithm: Image Processing Pipeline

The diagram below summarizes the pipeline to process images.

<img src="output_images/algo_diagram.png">

## Result

<a href="https://www.youtube.com/embed/ztvgLcQjkjc><img src="http://img.youtube.com/vi/ztvgLcQjkjc/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>


## How to run

1. Install [anaconda](https://www.continuum.io/downloads)
2. Install and activate the [carnd-term1](https://github.com/udacity/CarND-Term1-Starter-Kit) conda environment
4. Run ` $ jupyter notebook` on the directory containing the IPython notebook and open `Vehicle Detection and Tracking - Oscar Argueta.ipynb`
