import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Giving different arrays to handle colour points of different colours
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

# Define colors: Blue, Green, Red, Yellow
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0  # Default color is Blue

# Here is code for Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)      # Clear
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], 2)     # Blue
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], 2)     # Green
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], 2)     # Red
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], 2)     # Yellow

# Put text labels on the buttons
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

# Initialize a drawing flag
drawing = False

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame not read correctly

    x, y, c = frame.shape

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB for MediaPipe
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the color selection and clear buttons on the frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)       # Clear
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], 2)      # Blue
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], 2)      # Green
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], 2)      # Red
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], 2)      # Yellow

    # Put text labels on the buttons
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post-process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # Convert normalized landmarks to pixel coordinates
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on the frame
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Extract the coordinates of the index finger tip and thumb tip
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])

        # Draw a small circle at the index finger tip
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        # Debugging: Print the vertical distance between thumb and index finger
        # This helps in detecting gestures
        distance = center[1] - thumb[1]
        print(f"Vertical Distance (Index - Thumb): {distance}")

        # Gesture Detection
        if (thumb[1] - center[1] < 30):
            # Gesture to stop drawing or switch modes
            drawing = False

            # Initialize a new deque for the currently selected color only
            if colorIndex == 0:
                bpoints.append(deque(maxlen=512))
                blue_index += 1
            elif colorIndex == 1:
                gpoints.append(deque(maxlen=512))
                green_index += 1
            elif colorIndex == 2:
                rpoints.append(deque(maxlen=512))
                red_index += 1
            elif colorIndex == 3:
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

        elif center[1] <= 65:
            # User is selecting a color or clearing the canvas
            drawing = False  # Not drawing while selecting

            if 40 <= center[0] <= 140:  # Clear Button
                # Reset all points and indices
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                # Clear the paint canvas
                paintWindow[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
        else:
            # Actively drawing
            drawing = True
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    else:
        # When no hands are detected, stop drawing
        drawing = False
        # Optionally, you can add logic here to handle missing hands

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                # Draw the line on both the frame and the paint window
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Combine the webcam frame and the paint window
    # You can display them side by side or in separate windows
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
