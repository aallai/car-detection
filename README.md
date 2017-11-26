

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

The code for hog feature extraction is located in the extract_features function in train.py. I used the YUV color space after experimenting with a few color spaces. At first I also used color histogram features, but eventually dropped them while trying to reduce false positves. They didn't seem to make a huge difference to classifier accuracy.

Example YUV HOG features:

![alt text][image1]


#### 2. Explain how you settled on your final choice of HOG parameters.

I used classifier accuracy as the metric for my choice of model parameters. The final accuracy of the linear SVM is around 98%. At first I was using a kernel svm, but inference was very slow, so I switched to a linear svm. With parameter tweaking I was able to acheive the around same accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The train.py file implements loading the dataset and training the svm. I augment the negative training examples using code from the traffic sign classification project, and end up with twice as many non-car examples. My goal is to try to reduce the false positive rate by biasing towards negative predictions.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The generation of sliding windows is implemented by slide_window() in pipeline.py. The sizes/scales used are in process_image() in the same file. I spent a lot of time playing with different windows, the two biggest problems were false positives and slow runtime. I found that sticking to larger windows of the same aspect ratio as the training examples worked best.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The main thing I did to improve performance was to reject predictions with low confidence, generally anything below 0.8 or 0.9 probability of being a car. Along with not using small windows, this got rid of most false positives. Here are a few examples of bounding boxes without clustering:

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

I keep about 20 frames of history, and threshold out areas that were in less than 15 detections. This seems to reduce the wobbliness of the windows.

Here is a frame, the heatmap accumulated in the last 20 frames, and the final output.

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Currently the pipeline fails briefly for the white car, in an area around a sign, and when it gets far away. The first case could probably be fixed by improving the classifier. The second case can be improved by adding smaller detection windows. This blows up the runtime and also tends to add more false positives.

The pipeline is slow, at about 0.5 frames per second. This could be improved by precomputing HOG features for the whole image. When I profiled my code though, most of the time was spent running inference on all of the sliding windows. I might investigate parralelizing this. Another idea would be to use an approach like SSD or YOLO, that regresses a fixed number of bounding boxes in one forward pass of a neural net. This probably parralelizes well on a GPU.

