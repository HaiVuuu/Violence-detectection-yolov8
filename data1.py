from ultralytics import YOLO
import cv2
import numpy as np
model=YOLO('yolov8n-pose.pt')

# results=model(source='real_life_violence/Violence/V_1.mp4',show=True,conf=0.1)

cap=cv2.VideoCapture('real_life_violence/Violence/V_4.mp4')

fr_ls=[]
while True:
    ret,frame=cap.read()
    if ret:
        results=model(frame)
        res=results[0].keypoints.numpy().xy

        for people in res:
            for point in people:
                cv2.circle(frame, (int(point[0]),int(point[1])), 10, (0, 0, 255), cv2.FILLED)
            
        
        cv2.imshow('test',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()


