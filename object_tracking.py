import cv2
import numpy as np

cap=cv2.VideoCapture("vehicle_video.mp4")
# the while loop helps in getting all frames, 1ms between each frame
#object detection from stable camera
object_detector=cv2.createBackgroundSubtractorMOG2()
# this will extract all the moving object from the frame
while True: 
 ret, frame=cap.read()
 height, width, _=frame.shape
 #extract region of interest
 roi= frame[340:720,5:900]
 #height ans width for region which you can change later where actual detection and numbering will happen
 mask=object_detector.apply(roi)
 # mask is basically used to apply at all our frames, to roi actually the region that we have selected which make detectable objects such as vehicle white and other objects black
 contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 for cnt in contours:
  #calculate area and remove small ele
  area=cv2.contourArea(cnt)
  if area >100:
    cv2.drawContours(roi, [cnt], -1, (0,255,0),2)

 cv2.imshow("roi", roi)
 cv2.imshow("Frame", frame)
 cv2.imshow("Mask", mask)

 key=cv2.waitKey(1)
 if key == 27:
  break
 
cap.release()
cv2.destroyAllWindows()
# a function that keeps the window intact , so that the frame just doesn't goes off
