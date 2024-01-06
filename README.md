# ComputerVisionProjectEcam2023
## Noé TARDY

### Introduction
The aim of this project is to develop a system capable of recognizing hand gestures for the game Rock-Paper-Scissors using a camera. This system is implemented using Python and the OpenCV library. The challenge is to accurately detect the difference between a Rock, a paper, and a scissor made with the hand.

### Method
The program operates by continuously capturing frames from the webcam, applying a skin color filter to isolate the hand and the rest of the exposed body. By comparing consecutive frames, it identifies the moving part of the image, presumed to be the hand, and applies morphological operations to enhance the mask. The program then detects contours and computes the convex hull around the largest contour, assumed to be the hand shape. It calculates the number of corners, the area, and the average distance between corners of the hull over the last second to characterize the hand gesture. These characteristics are compared against predefined thresholds to classify the gesture as Rock, Paper, or Scissors. The program updates and displays this information in real-time.

A convex hull is a concept in computational geometry that refers to the smallest convex set that encloses a given set of points. In simpler terms, if you imagine stretching a rubber band so that it encompasses all the points and then releasing it, the shape that the rubber band takes is the convex hull.

### Failed Attempts
Initially, I attempted to use Hough Circle and Hough Line detection techniques for hand recognition. However, these methods did not yield satisfactory results due to their inability to effectively isolate and interpret complex hand gestures.

Another approach was template matching using the build-in OpenCV function. While this method showed some promise, running multiple instances for different hand positions required extensive computational resources, making it impractical for real-time applications.

### Implementation
1. **Capture Video:** The script captures a video frame from the webcam.
2. **Color Conversion and Skin Filter:** The captured BGR image is converted to the HSV (Hue, Saturation, Value) color space. This conversion facilitates the application of a skin color filter.

    ![Color filter mask](#)
   
3. **Movement Detection:** By computing the difference between consecutive frames, the script effectively isolates the moving parts in the video, primarily the hand.
4. **Mask Post-Treatment:** The mask undergoes noise reduction and dilation to enhance the hand's representation in the frame.
5. **Contour Detection and Convex Hull:** The script detects contours in the masked image. The largest contour is assumed to represent the hand, and a convex hull is drawn around it. This hull helps in estimating the shape of the hand.
6. **Gesture Differentiation:** The system classifies the gesture as Rock, Paper, or Scissors based on the average measurements of the hull's area, the number of corners, and the average distance between corners over the last second.

    These measurements are matched against predefined ranges determined through experimental observations.

    | Classification Criteria | Paper  | Rock  | Scissors |
    |-------------------------|-------|-------|----------|
    | Corners                 | 15-17 | 15-18 | 13-15    |
    | Area                    | 125-140 | 60-80 | 90-120  |
    | Distance of Corners     | 240 - 270 | 190-210 | 250-270|

### Results
An accompanying video demonstrates the system in action, displaying real-time data on the area, corner count, and corner distances, along with the identified gesture (Rock, Paper, or Scissors).

The program runs at approximately 30 frames per second on my old MacBook. The detection is fairly accurate and usually takes less than a second to converge to the correct gesture.

### Challenges and Problems
- **Lighting Variation:** The system's performance is somewhat dependent on lighting conditions, which can affect skin color detection.
- **Skin Tone Diversity:** Currently, the skin filter is calibrated for lighter skin tones, limiting its effectiveness across a diverse range of skin colors.
- **Environmental Constraints:** The hand must be in motion for accurate detection. Additionally, the system requires a stationary computer and a stable background to function optimally.
- **Camera Resolution:** It doesn’t work with different camera resolutions since the area will not be the same.

### Conclusion
This project was both challenging and enjoyable. The complexity of real-time gesture recognition was greater than initially anticipated. However, working through these challenges was a valuable learning experience. Future improvements could focus on enhancing the system's robustness to diverse skin tones and varying environmental conditions. Modern deep learning techniques offer more robust and adaptable solutions, effectively handling a wider range of conditions and complexities.

### Sources
For understanding the concept of convex hull, I referred to an insightful article on Medium titled "Contours and Convex Hull in OpenCV Python" ([Link](https://medium.com/analytics-vidhya/contours-and-convex-hull-in-opencv-python-d7503f6651bc)).

GPT-4 played a role in the development of the algorithm, particularly in identifying the closest match for gesture recognition. It introduced me to the Python deque function, enabling the calculation of average values over the last few seconds. Its ability to comprehend, debug, and optimize code proved invaluable, significantly reducing development time.
