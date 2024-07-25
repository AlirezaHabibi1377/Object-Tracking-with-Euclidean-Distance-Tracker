# Object Tracking with Euclidean Distance Tracker

This Python script demonstrates object tracking using the Euclidean Distance Tracker and OpenCV. It processes a video to detect and track objects in real-time.

# Features
1. Object Detection: Utilizes background subtraction to detect moving objects.
2. Object Tracking: Uses the Euclidean Distance Tracker to track detected objects across frames.
3. Visualization: Displays the detected objects and their IDs in real-time.

# How It Works
### 1. Initialize: Creates an instance of the Euclidean Distance Tracker and sets up video capture.
### 2. Object Detection: Applies background subtraction to extract regions of interest (ROIs) and detects objects based on contours.
### 3. Object Tracking: Updates object positions and assigns IDs using the tracker.
### 4. Display: Shows the tracked objects, their IDs, and the original video frames in separate windows.

# Prerequisites
### Python 3.9
### OpenCV library (opencv-python)
