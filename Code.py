import cv2
import numpy as np
from collections import deque
from math import sqrt

# Global variables to store the previous values
prev_num_corners = 0
prev_area = 0
prev_avg_dist = 0
prev_closest_case = "Unknown"
def distance(pt1, pt2):
    # Calculate Euclidean distance between two points
    return sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

def get_closest_case(avg_corners, avg_area, avg_dist):
    # Define the cases with their corner, area, and distance ranges
    cases = {
        'Paper': {'corners': (15, 18), 'area': (120, 140), 'dist': (240, 270)},
        'Rock': {'corners': (15, 18), 'area': (60, 80), 'dist': (190, 210)},
        'Scissors': {'corners': (13, 15), 'area': (90, 115), 'dist': (250, 270)}
    }

    closest_case = "Unknown"
    min_diff = float('inf')

    # Compare the average values with each case and find the closest one
    for case, ranges in cases.items():
        diff = abs(avg_corners - sum(ranges['corners']) / 2) + \
               abs(avg_area - sum(ranges['area']) / 2) + \
               abs(avg_dist - sum(ranges['dist']) / 2)
        print(diff)
        if diff < min_diff:
            min_diff = diff
            closest_case = case

    # Define a threshold for when to consider the match as "Unknown"
    threshold = 80 
    if min_diff > threshold:
        closest_case = "Unknown"

    return closest_case

def detect():
    global prev_num_corners, prev_area, prev_avg_dist, prev_closest_case

    # Start capturing from the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Initialize 'prev_frame' as None
    prev_frame = None

    # Queues to store the last second's worth of data (assuming 30 fps)
    corner_history = deque(maxlen=30)
    area_history = deque(maxlen=30)
    dist_history = deque(maxlen=30)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Convert current frame to grayscale for motion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If 'prev_frame' is not None, calculate the absolute difference
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, gray_frame)

            # Lower the threshold to make the detection more sensitive to movement
            _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)

            # Define the range of skin color in HSV
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # Create a mask with the specified range
            skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

            # Combine the skin mask with the motion mask
            motion_skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=thresh)

            # Use morphological operations to remove noise and close gaps
            kernel = np.ones((5,5), np.uint8)
            kernel2 = np.ones((10,10), np.uint8)
            motion_skin_mask = cv2.morphologyEx(motion_skin_mask, cv2.MORPH_OPEN, kernel)
            motion_skin_mask = cv2.dilate(motion_skin_mask, kernel, iterations=2)
            motion_skin_mask = cv2.morphologyEx(motion_skin_mask, cv2.MORPH_CLOSE, kernel2)

            # Find contours
            contours, _ = cv2.findContours(motion_skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour by area
            if contours:
                max_contour = max(contours, key=cv2.contourArea)

                # Find the convex hull for the largest contour
                hull = cv2.convexHull(max_contour)

                # Draw the convex hull on the frame
                cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)

                # Calculate the number of corners and area
                num_corners = len(hull)
                area = cv2.contourArea(hull)/1000
                corner_history.append(num_corners)
                area_history.append(area)

                # Calculate average distance between corners if possible
                avg_dist = 0
                if num_corners > 1:
                    distances = [distance(hull[i][0], hull[j][0]) for i in range(len(hull)) for j in range(i+1, len(hull))]
                    avg_dist = sum(distances) / len(distances)
                    dist_history.append(avg_dist)

        # Update 'prev_frame' for the next iteration
        prev_frame = gray_frame

        # Calculate the averages if enough data is available
        if len(corner_history) == corner_history.maxlen:
            avg_corners = sum(corner_history) / len(corner_history)
            avg_area = sum(area_history) / len(area_history)
            avg_dist = sum(dist_history) / len(dist_history) if dist_history else 0

            # Determine the closest case
            closest_case = get_closest_case(avg_corners, avg_area, avg_dist)
            prev_closest_case = closest_case

            # Update the global variables
            prev_num_corners = avg_corners
            prev_area = avg_area
            prev_avg_dist = avg_dist

        # Display the previous values
        cv2.putText(frame, f'Corners: {prev_num_corners:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Area: {prev_area:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if prev_avg_dist:
            cv2.putText(frame, f'Avg Dist: {prev_avg_dist:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Closest Case: {prev_closest_case}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Result', frame)
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

detect()





