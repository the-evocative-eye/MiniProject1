import cv2
import numpy as np
from object_detection import *

tracker=EuclideanDistTracker()

cap=cv2.VideoCapture("vehicle_video.mp4")
# the while loop helps in getting all frames, 1ms between each frame
#object detection from stable camera
object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
#lower the value of varThreshold , more the detection but more num of false positives, that's why we are taking middle value
# this will extract all the moving object from the frame, object detection algorithm(mog2)
while True: 
 ret, frame=cap.read()
 height, width, _=frame.shape
 #extract region of interest
 roi= frame[340:720,5:900]
 #height ans width for region which you can change later where actual detection and numbering will happen
 mask=object_detector.apply(roi)
 _, mask=cv2.threshold(mask, 254,255,cv2.THRESH_BINARY)
 # mask is basically used to apply at all our frames, to roi actually the region that we have selected which make detectable objects such as vehicle white and other objects black
 contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 detections=[]
 for cnt in contours:
  #calculate area and remove small ele
  area=cv2.contourArea(cnt)
  if area > 1300:
    # cv2.drawContours(roi, [cnt], -1, (0,255,0),2)
    x,y,w,h=cv2.boundingRect(cnt)


    detections.append([x,y,w,h])
    #object tracking
    boxes_ids=tracker.update(detections)
    for box_id in boxes_ids:
     x,y,w,h,id=box_id
     cv2.putText(roi,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
     cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)

 cv2.imshow("roi", roi)
 cv2.imshow("Frame", frame)
 cv2.imshow("Mask", mask)

 key=cv2.waitKey(15)
 if key == 27:
  break
 
cap.release()
cv2.destroyAllWindows()
# a function that keeps the window intact , so that the frame just doesn't goes off
