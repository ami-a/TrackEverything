# TrackEverything

This project is an open-source package built in Python, it uses and combine the data form object detection models, classification models, tracking algorithms and statistics-based decision making. The project allows you to take any detection/classification models from any Python library like TensorFlow or PyTorch and add to them tracking algorithms and increase the accuracy using statistical data gathered from multiple frames.
<br>
Contributions to the codebase are welcome and I would love to hear back from
you if you find this package useful.
## How does it work
I recommend jumping to [this](#the-pipeline) part first to understand the pipeline and the methods used.

## Installation & Requirements [![Python 3.7+](https://img.shields.io/badge/Python-3.7+-3776AB)](https://www.python.org/downloads/release/python-370/)  [![OpenCv 3.5+](https://img.shields.io/badge/OpenCv-3.5+-3fd4d4)]()
You can easily install the package with the Python Package Installer pip.
I used Python 3.8 but the package should work for Python 3.7+, additional requirements like NumPy will be checked and installed automatically.

```bash
# upgrade pip
python -m pip install --upgrade pip
# TrackEverything
python -m pip install TrackEverything
```
## How to Start

### Available Examples

I made two different repositories that demonstrate the use of this package.

* [Cop Detection](https://github.com/ami-a/CopDetection) - An example of using a famous object detection model and custom classification model to detect with high accuracy, law-informant personals.
* [Mask Detection](https://github.com/ami-a/MaskDetection) - Few different examples of using the package with head detection/face detection/face detection+classification models, to find and classify with high accuracy, persons with or without a mask.

### Basic Steps

The main class is called a detector (`Detector`), you first need to define it's parameters.
* `DetectionVars`- contains the detection model itself as well as interpolation methods (you can use a model the dose both).
* `ClassificationVars` - contains the classification model (if exist) as well as interpolation methods.
* `InspectorVars` - contains the logic as well as the statistical parameters like tracking type and statistics methods like moving average. (The default value will not use previous data)
* `VisualizationVars` - contains some parameters for the drawing on the frames if needed.

Once your detector is all set, you can use the `update(frame)` method to update all the data according to the new frame.
If you want to add the result to the frame, simply use the `draw_visualization(frame)` method to add bounding boxes and text to the frame.

## More Options

### Pick A Different Tracker Type
I use in this package tracker objects from the OpenCV library, in the `InspectorVars` class you can choose different type of trackers, the default tracker type is [CSRT](https://docs.opencv.org/3.4/d2/da2/classcv_1_1TrackerCSRT.html) (A [Discriminative Correlation Filter Tracker with Channel and Spatial Reliability](https://arxiv.org/abs/1611.08461)).

<p align="center"><img src="images/charts/csr_dcf.png" width="506" height="446"/><br>Overview of the CSR-DCF approach. An automatically estimated spatial reliability map restricts the correlation filter to the parts suitable for tracking (top) improving localization within a larger search region and performance for irregularly shaped objects. Channel reliability weights calculated in the constrained optimization
step of the correlation filter learning reduce the noise of the weight-averaged filter response (bottom).</p>

But there are many more trackers types in OpenCV that you can choose from, here is a summary by Adrian Rosebrock: 
* **BOOSTING Tracker**: Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost), but like Haar cascades, is over a  decade old. This tracker is slow and doesn’t work very well. Interesting only for legacy reasons and comparing other algorithms. (minimum OpenCV 3.0.0)
* **MIL Tracker**: Better accuracy than BOOSTING tracker but does a poor job of reporting failure. (minimum OpenCV 3.0.0)
* **KCF Tracker**: Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well. (minimum OpenCV 3.1.0)
* **CSRT Tracker**: Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower. (minimum OpenCV 3.4.2)
* **MedianFlow Tracker**: Does a nice job reporting failures; however, if there is too large of a jump in motion, such as fast moving objects, or objects that change quickly in their appearance, the model will fail. (minimum OpenCV 3.0.0)
* **TLD Tracker**: I’m not sure if there is a problem with the OpenCV implementation of the TLD tracker or the actual algorithm itself, but the TLD tracker was incredibly prone to false-positives. I do not recommend using this OpenCV object tracker. (minimum OpenCV 3.0.0)
* **MOSSE Tracker**: Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed. (minimum OpenCV 3.4.1)
* **GOTURN Tracker**: The only deep learning-based object detector included in OpenCV. It requires additional model files to run. My initial experiments showed it was a bit of a pain to use even though it reportedly handles viewing changes well. (minimum OpenCV 3.2.0)

### Pick Different Statistical Method - StatisticalCalculator
Inside the `InspectorVars` class you can insert a `StatisticalCalculator` object, this class currently contains several different statistical methods.
* **Non** -No statistical information is saved.
* **CMA - Cumulative Moving Average** - The data arrive in an ordered datum stream, and the user would like to get the average of all of the data up until the current datum point.
* **FMA - Finite Moving Average** - The result is the unweighted mean of the previous n data.
* **EMA - Exponential Moving Average** - It is a first-order infinite impulse response filter that applies weighting factors which decrease exponentially.

This are just some basic methods and you can add many more.

### Others
There are many more options in this package, you can use the built in Non-Max Suppressions on your models, you can give each classification category a different weight in the statistics.
For example - you can use a model to define a person's mood by it's face using head detection and a classification model that gives a back a category of 0 if it dose not have high enough score (for example if the persons is with it's back to the camera). You can then set the impact (0.0-1.0) of category 0 to be very low, and so when the person turns around the data on him is saved and is not overwritten.

## Future Improvements
* Add support for multiple cameras
* Add an option to run the entire project on the Raspberry Pi using TensorFlow Lite

## The Pipeline

The pipeline starts by receiving a series of images (frames) and outputs a list of tracker objects that contains the objects detected and the probability of them being in a class.
<p align="center"><img src="images/charts/pro_flow.png" width=650 height=424></p>

## Breaking it Down to 5 Steps

### 1st Step - Get All Detections in Current Frame 

First, we take the frame and passe it through an object detection model, we can use any Python model, then filter out redundant overlapping detections using the Non-maximum Suppression (NMS) method and add all of the detection to the `detections` list.

### 2nd Step - Get Classification Probabilities for the Detected Objects

After we have the detections from step 1, we put them through a classification model to determine the probability of them being in a certain class (if no classification model is supplied the classification is applied during the previous step). We do this by cropping the frame to the object bounding box and then pass it through the classification model. We add this data as a vector of probabilities to each of the detection in the `detections` list.

### 3rd Step - Updated the Trackers Object List

We have a list of `trackers` object which is a class that contains among other things an OpenCV tracker object, unique ID, previous statistics about this ID and indicators for the accuracy of this tracker. In the first frame, this `trackers` list is empty and then in step 4, it's being filled with new trackers matching the detected objects. If the `trackers` list is not empty, in this step we update the trackers positions using the current frame and dispose of failed trackers.

### 4th Step - Matching Detection with Trackers

Using intersection over union (IOU) of a tracker bounding box and detection bounding box as a metric. We solve the linear sum assignment problem (also known as minimum weight matching in bipartite graphs) for the IOU matrix using the Hungarian algorithm (also known as Munkres algorithm). The machine learning package `scipy` has a build-in utility function that implements the Hungarian algorithm.
```bash
matched_idx = linear_sum_assignment(-IOU_mat)
```
The linear_sum_assignment function by default minimizes the cost, so we need to reverse the sign of IOU matrix for maximization.<br>
The result will look like this:<p align="center"><img src="images/charts/detection_track_match.png" width=548 height=426></p>
For each unmatched detector, we create a new tracker with the detector's data, for the unmatched trackers we update the accuracy indicators for the tracker and remove any that are way off. For the matched ones, we update the tracker position to the more accurate detection box, we get the class data and average it with the previous 15 data points of the tracker (you can change that by setting the `num_avg` var).

### 5th Step - Decide What to Do

After step 4 the `Trackers` list is up to date with all the statistical and current data. The tracker class has a method to return the current classifications and confidence of those scores, we then update the detectors and iterate through them. A detector with low confidence score probably came from a tracker with not enough data (need at least `num_avg` data point to get the maximum score) or the detection is poor, we mark those in orange. A detector with a high enough confidence score will be green if it's not a cop, and red/blue if it is. The trackers that are not covered with detection boxes will show in cyan.
<p align="center"><img src="images/screens/vid_08.png" width=564 height=383></p>
<p align="center"><img src="images/screens/vid_03.png" width=564 height=383></p>

## Results

I only tested it on some videos I found online but the results are pretty good for cops and ok for parking officers. It requires more work and maybe a decent cop dataset to take the cop model further and into more countries.





