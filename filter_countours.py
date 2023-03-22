import cv2
import numpy as np

# Create a VideoCapture object
cap = cv2.VideoCapture('Videos/video35.mp4')
#cap = cv2.VideoCapture(0)

frame_count = 0

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_red = np.array([165, 70, 50])
        upper_red = np.array([195, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  #Umbral del area a detectar
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
                #cv2.imwrite('frameImage/frame_{}.jpg'.format(frame_count),frame)
                #frame_count += 1

        cv2.imshow('Original', frame)
        cv2.imshow('res', res)
       
        delay = int(1000/fps)
        cv2.waitKey(delay)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break


# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()