## Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier: Linear SVM classifier
* Applying a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/01_RawImage01.png
[image2]: ./output_images/01_RawImage02.png
[image3]: ./output_images/02_explore.png
[image4]: ./output_images/03_hog_car_01gray.png
[image5]: ./output_images/03_hog_notcar_01gray.png
[image6]: ./output_images/03_hog_car_02rgb.png
[image7]: ./output_images/03_hog_notcar_02rgb.png
[image8]: ./output_images/03_hog_car_03y.png
[image9]: ./output_images/03_hog_notcar_03y.png
[image10]: ./output_images/04_detection_heatmap01.png
[image11]: ./output_images/04_detection_heatmap02.png
[image12]: ./output_images/04_detection_heatmap03.png
[image13]: ./output_images/04_detection_heatmap04.png
[image14]: ./output_images/05_vid_frame01.png
[image15]: ./output_images/05_vid_frame02.png
[image16]: ./output_images/05_vid_frame03.png
[image17]: ./output_images/05_vid_frame04.png
[video14]: ./project_video_output.mp4



## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) 
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!
### Data Exploration (HOG)
To get started, I just look at the test-images first, to understand what I am dealing with.
Here are 2 of the test images:

![Explore test image][image1]
![Explore test image][image2]

Here are stats of the training set:

```
Your function returned a count of 8792  cars and 8968  non-cars
of size:  (64, 64, 3)  and data type: float32
```

And, here is a random car image, and random not-car image:

![Explore test image][image3]

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I first implemented just extraction of Hog features, on a grayscaled (single-channel) image, in the 4th cell of the iPython notebook, in the function `grayscale_and_hog()`
Here is an example of its output, one for a car, and the next for a non-car image.

![car hog][image4]
![Not car hog][image5]

I experiemented with modifying the HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. 
First I modified cells_per_block to get more granularity, but that gave me a performance penalty, i.e. it ran very slow. 
So to get a better idea of which paramters to tweak, I actually tried running my classifier to see what gave me most accuracy, and a still ran in a reasonable amount of time.


#### 2. Explain how you settled on your final choice of HOG parameters.

After getting the grayscale image's hog-features workings, I tried it with different color-spaces.
Here are each of the 3 color channels for RGB color-space, for both car and not-car images.

Hog-features of individual channels of RGB, car image:
![car hog][image6]


Hog-features of individual channels of RGB, not-car image:
![Not car hog][image7]

I tried other color spaces, as well, but finally settled on YCrCb color-space.
Thr reason why I settled on YCrCb, is because visually it seemed to provide crisper resolution of features. A clear distinction between the other hog-features and YCrCb is that in YCrCb color-space, we are able to see the rear-windshield more clearly, as evidenced by the images below. 
Similarly the non-car image's lane markers are also more pronounced.
This led me to believ that the YCrCb color-space was ideal for this project.

For indidivual paramters such as `pixels_per_cell`, I also experimented with other values, such as `6,6`, but I soon realized the values in the lecture code worked best, so I stuck with those.

Here are each of the 3 color channels for YCrCb color-space, for both car and not-car images.

Hog-features of individual channels of YCrCb, car image:
![car hog][image8]


Hog-features of individual channels of YCrCb, not-car image:
![Not car hog][image9]


Finally, I integrated the hog-features with binned and histograms of color in the 5th cell of the iPython notebook, in: `extract_features()`. This function is very similar to the one from lecture. It, in turn, calls the following:

* `bin_spatial()` for spatial binning
* `color_hist()` for histogram of colors, and finally
* `get_hog_features()` to get the hog features

Then it appends them to build the full feature-vector.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a `LinearSVC()`.
The code is in the 6th cell of the iPython notebook, starting with: `# Extract Features and Train Classifier - Linear SVM`.

Essentially, the classifier extracts all features, as described above, of both the car and not-car images, normalizes them using a `StandardScaler()`, and then it classifies them to be car OR not-car.

I used 20% (0.2) data split for testig the qualifier.

I tried this with all the color-spaces too.

Finally, I got the best performance with YCrCb color-space and achieved an accuracy of 98.56% (0.9856)


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have implemented a sliding window search in the 7th cell of my iPython notebook, starting with: `# A single function that can extract features using hog sub-sampling and make predictions`. It is very similar to the `find_cars()` method from the lectures. 

However, I made the following 4 major changes:

* Heatmap: I implemented heatmap and thresholding on top of it, which can also be found in the same cell of my iPython notebook, in `find_cars_and_draw()` method. With experimentation, I determined the best threshold to be 1.5
* Scale: I experiemented with the scale values between 1.0 - 2.0, with steps of 0.25. A scale of 1.0 would capture the much further ahead cars well, and a scale of 2.0 was too big, and it would capture cars nearby well. So implemented a for loop to go through scales of: 1.0, 1.25, and 1.75. This gave me the best results to capture all car images.
* Overlap: Initially, I left the overlap at 2 cells per step. But the boxes were too distant and the heatmap was light. Reducing this to 1 did the trick, so I stuck with that.
* Distance threshold: In addition, to remove more false-positives, I also implemented test-distance threshold, which would basically take the output of the decision-function of the LinearSVC and determine how far away from the separation hyper-plane the result is. With some experimentation on the test images and the test-video, which would actually remove the false-positives on the road, I settled on a threshold of 1.2

The output images are in the next section.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As described above, ultimately, I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images, from the test-images.

The top-left image is the original image.

The top-right is the heat-map, with the threshold applied.

The bottom-left is the image with cars detected, but without the heat-map threhold applied.

The bottom-right is the final image with cars detected, with overlapped bounding-boxes combined, and the heat-map threshold applied.

Image 1:
![output image 1][image10]

Image 2:
![output image 1][image11]

Image 3:
![output image 1][image12]

Image 4:
![output image 1][image13]


To improve the performance of the classifier, I used the method described in the lectures, where we only have to extract the Hog features once. This is very similar to the `find_cars()`  method described in the in lectures, and it is in my `find_cars()` method. It extracts hog features once, for each of a small set of predetermined window size, which is defined by the  scale, and then it sub-samples to get all of its overlaying windows.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The video code has been implemented in 8th (last) cell of my iPython notebook starting with `# Run it on the video`.

* False-positives: To eliminate the false-positives, I deployed a summation of heatmaps and thresholded it higher than the images in the sections above. I recorded the positions of positive detections in each frame of the video. I kept a window (queue) of 10 heat-maps. After each subsequent frame, I added the heatmap from the previous frame. Finally, with experimentation, I determined the best summed threshold to be 25.0, and I applied it to this summed up heatmap. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected, and combined them as described below.

* Combining overlapping bounding boxes: This is implemented in `draw_labeled_bboxes()` function. It just takes the extreme limits of the heatmap bounding boxes, and merges into 1 rectangular bounding box.

### Here are six frames, their corresponding thresholded heatmaps, and also their resulting bounding boxes drawn:

Here you can see some ephemeral false-positives get eliminated as the frames progress forward. 

Intial Frame 1:

Frame 1:
![vid frame1][image14]

Second Frame 2:

Frame 2:
![vid frame2][image15]

We can see the false-positive here in Frame 3, but it does not appear in the final image (at the bottom-right) because it gets removed by heatmap summation thresholding.

Frame 3:
![vid frame3][image16]

Finally, in frame 4, it false-positives do not appear again.

Frame 4:
![vid frame4][image17]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced quite a few problems:

* BGR vs RGB: My intial SVM prediction was almost completely wrong. This was because I was reading images with cv2, which reads as BGR, as opposed to my find_cars() which was using mping, which reads in RGB. So once I corrected that, my pipeline started working better.
* Using a single scale of 1.75: I experiemented with the scale values between 1.0 - 2.0. A scale of 1.75 gave me OK results for the test-images. But on the video with cars near and far it was bad, so I decided to implement multiple scales.
* Overlap of 2: Initially, I left the overlap at 2 cells per step. But the boxes were too distant and the heatmap was light. Reducing this to 1 did the trick, so I stuck with that.
* Not having a distance threshold: In a final push to eliminate false-positives, I implemented a thresholding on distance from separation hyperplane. This eliminated more false-positives.

Improvements:

* I still have problems of some jitter especially near the bridge section, with lots of shadows around it. I would really like to eliminate it. I think working with an additional color-space, which is not affected by objects in different shades of light, could possibly eliminate it. Secondly, passing in all color-spaces as inputs to a Deep-learning pipeline could also solve it, because it would adjust the weights of different color spaces by itself.

