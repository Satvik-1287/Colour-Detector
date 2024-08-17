import cv2
import numpy as np

# Define the color ranges and their names in HSV color space
COLOR_RANGES = [
    {"name": "red", "lower": np.array([0, 120, 70]), "upper": np.array([10, 255, 255])},
    {"name": "red", "lower": np.array([170, 120, 70]), "upper": np.array([180, 255, 255])},
    {"name": "yellow", "lower": np.array([25, 120, 70]), "upper": np.array([35, 255, 255])},
    {"name": "green", "lower": np.array([40, 120, 70]), "upper": np.array([80, 255, 255])},
    {"name": "cyan", "lower": np.array([80, 120, 70]), "upper": np.array([100, 255, 255])},
    {"name": "blue", "lower": np.array([100, 120, 70]), "upper": np.array([130, 255, 255])},
    {"name": "magenta", "lower": np.array([130, 120, 70]), "upper": np.array([170, 255, 255])},
]

# Function to draw bounding boxes and color names on the frame
def draw_bounding_box(frame, name, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for color in COLOR_RANGES:
        mask = cv2.inRange(hsv_frame, color["lower"], color["upper"])
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes and color names
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small contours
                x, y, w, h = cv2.boundingRect(contour)
                draw_bounding_box(frame, color["name"], x, y, w, h)
    
    # Display the frame
    cv2.imshow('Multi-Color Detector', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
