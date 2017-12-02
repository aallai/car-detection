

**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./output_images/hog.png
[image2]: ./output_images/detections3.jpg
[image3]: ./output_video.gif
[image4]: ./output_images/detections5.jpg

[image5]: ./output_images/original1000.jpg
[image6]: ./output_images/heatmap1000.jpg
[image7]: ./output_images/final1000.jpg

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for hog feature extraction is located in the extract_features function in train.py. I used the YUV color space after experimenting with a few color spaces. I also used spatial features on a 32x32 scaled version of the image.

Example YUV HOG features:

![alt text][image1]


#### 2. Explain how you settled on your final choice of HOG parameters.

I used classifier accuracy to guide my choice initially. I had poor performance on the section of video where the white car passes the green road sign, so I cropped a few frames from that section and manually insepected the performance on those images.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The train.py file implements loading the dataset and training the svm. I augment the negative training examples using code from the traffic sign classification project, and end up with 4x non-car examples. I also mined some negative examples from a few difficult video frames. My goal was to reduce false positives.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented by slide_window() in pipeline.py. The scales/ROIs are defined in process_image() in the same file. I spent a lot of time playing with different windows, the two biggest problems were false positives and slow runtime. For false positives, improving the classifier and limiting the ROI for small windows was the most effective. To address slow runtime, I compute the HOG features once for every scale, and convolve the classifier across the HOG image, as suggested in the lectures.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

False positives were the main problem with my linear SVM classifier. To improve performance, I reject low-confidence predictions, augment the negative examples, use negative mining, and use a higher C parameter for negative examples. Here are a few examples of bounding boxes without clustering:

![alt text][image2]

Good detections.

![alt text][image4]

Example with a false positive.


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_video.mp4) (not available on github).

Here is a gif for the github readme:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used the heatmap method from the lectures. The code in is draw_bounding_boxes() in pipeline.py. Initially I tried to use the groupRectangles() function from opencv, but the results were not as nice. The main issue was that my sliding windows were too small to fully enclose the cars. It also tended to cluster rectangles from two close-by cars together, and output a rectangle halfway between the two. The heatmap method seems to segment them better. It also "smears" small detection windows across the whole car, so you tend to get a large enough bounding box even if your windows are smaller than the car.

I keep about 12 frames of history, and threshold out pixels in less than 20 detections.

Here is a frame, a heatmap showing the accumulated detections, and the final output.

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline has a few brief false positive windows. Using a better classifier might help there.

The pipeline is not fast enough for real-time use. Running the classifier on all the windows is parallelizable, and that could yield a good performance improvement. This seems like something that could be benefit from a GPU implementation.

