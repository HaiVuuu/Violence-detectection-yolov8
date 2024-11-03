import cv2
import mediapipe as mdp
import pandas as pd

cap=cv2.VideoCapture('kick/2.mp4')

mdp_pose=mdp.solutions.pose
pose=mdp_pose.Pose()
mdp_draw=mdp.solutions.drawing_utils

def process_timestep(result):
    print(result.pose_landmarks.landmark)
    dim=[]
    for i in result.pose_landmarks.landmark:
        dim.append(i.x)
        dim.append(i.y)
        dim.append(i.z)
        dim.append(i.visibility)
    return dim

def draw_landmark(mdp_draw,result,img):
    mdp_draw.draw_landmarks(img, result.pose_landmarks, mdp_pose.POSE_CONNECTIONS)

    for id, lm in enumerate(result.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


label='KICK'
lm_list=[]
while True:
    ret,frame=cap.read()
    if ret==True:
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        result=pose.process(frame_rgb)

        if result.pose_landmarks:

            lm=process_timestep(result)
            lm_list.append(lm)

            frame=draw_landmark(mdp_draw,result,frame)

        cv2.imshow('violence',frame)
    else:
        break
    k=cv2.waitKey(1)
    if k==ord('k'):
        break
cap.release()
cv2.destroyAllWindows()
df=pd.DataFrame(lm_list)
df.to_csv(label+'.csv')
