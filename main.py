import cv2
from tracker import *

# Create tracker object to keep track of the detected objects
tracker = EuclideanDistTracker()

# Path to the video file
highway = "highway.mp4"

# Capture the video from the file
cap = cv2.VideoCapture(highway)

# Object detection from a stable camera using Background Subtraction
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read() # Read a frame from the video
    if not ret:
        break  # Break the loop if there are no more frames
    
    height, width, _ = frame.shape # Get the dimensions of the frame

    # Define Region of Interest (ROI) in the frame
    roi = frame[340: 720,500: 800]

    # 1. Object Detection
    # Apply the background subtractor to the ROI
    mask = object_detector.apply(roi)

    # Apply a binary threshold to the mask to get a binary image
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = [] # List to store detected bounding boxes
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            # Get the bounding box coordinates for the contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Add the bounding box to the detections list
            detections.append([x, y, w, h])

    # 2. Object Tracking
    # Update the tracker with the new detections
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        # Draw the object ID above the bounding box
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # Draw the bounding box around the object
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the ROI, the original frame, and the binary mask
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Wait for 30 milliseconds and check if the user pressed the 'Esc' key
    key = cv2.waitKey(30)
    if key == 27:
        break # Exit the loop if the 'Esc' key is pressed

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
