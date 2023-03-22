
"""
import cv2
import numpy as np

# Define the lower and upper bounds of the green color in RGB space
lower_green = np.array([0, 128, 0])
upper_green = np.array([50, 255, 50])

# Open the video file
cap = cv2.VideoCapture('output4.mp4')

while cap.isOpened():
	# Read each frame of the video
	ret, frame = cap.read()

	if ret:
		# Apply the green color filter to the frame
		mask = cv2.inRange(frame, lower_green, upper_green)
		green_object = cv2.bitwise_and(frame, frame, mask=mask)

		# Show the original frame and the green object
		cv2.imshow('Original', frame)
		cv2.imshow('Green Object', green_object)

		# Press 'q' to exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
"""

import cv2
import os
import numpy as np

# Create a VideoCapture object
cap = cv2.VideoCapture('../samples1/video20.mp4')
#cap = cv2.VideoCapture(0)


# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
print(os.getcwd())


while True:
	# Capture frame-by-frame
	ret, frame = cap.read()
	# print(frame)
	
	if not ret:
		break
	# Apply a Gaussian blur to the frame
	blurred = cv2.GaussianBlur(frame, (21, 21), 0)

	# Convert the frame to HSV color space
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# Define a range of green color in HSV
	lower_green = np.array([165, 70, 50])
	upper_green = np.array([195, 255, 255])

	# Threshold the HSV image to get only green colors
	mask = cv2.inRange(hsv, lower_green, upper_green)

	# Apply the mask to the original frame
	res = cv2.bitwise_and(frame, frame, mask=mask)

	# Check if the frame was successfully captured
	if ret:
		# Display the resulting frame
		cv2.imshow('Original', frame)
		cv2.imshow('frame', res)


	# Limit the frame rate to 25 frames per second
	delay = int(1000/fps)
	cv2.waitKey(delay)

	# Quit the program when 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
		break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
