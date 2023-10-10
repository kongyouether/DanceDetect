from ultralytics import YOLO
import cv2
import math
import os
import glob
import numpy as np


from read import drawpoint
import json
# Load a model

def detect (thecap, index):
    success, frame = thecap.read()
    if index == 0:
        frame = cv2.flip(frame, 1)
    # cv2.imwrite("test.jpg", frame)
    results = model(frame, imgsz=256)
    annotated_frame = results[0].plot() # or .show()
    # print(results[0].tojson('data.json'))
    # 将结果保存到json文件中
    resultsdata = results[0].tojson('data.json')
    # print("len:",len(resultsdata))
    # 将resultsdata中的数据存储到txt文件中
    if len(resultsdata) > 5:
        with open('data' + str(index) + '.txt', 'w') as f:
            f.write(resultsdata)
        # 将txt文件中的数据读取出来
        angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip, frame = drawpoint(index,frame)
    else:
        print("没有检测到人体")
        cv2.imshow("img"+ str(index), frame)
        angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip = 0, 0, 0, 0, 0, 0, 0, 0
    # cv2.imshow("YOLOv8 pose inference", annotated_frame)

    return angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip, frame


model = YOLO('yolov8s-pose.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained

video_path0 = 0 # 'path/to/video.mp4 or 0 for webcam
cap0 = cv2.VideoCapture(video_path0)
video_path1 = 'dance.mp4' # 'path/to/video.mp4 or 0 for webcam
cap1 = cv2.VideoCapture(video_path1)

#
while cap0.isOpened():
    angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip,img = detect(cap0, 0)
    angle_leftknee1, angle_rightknee1, angle_rightelbow1, angle_leftelbow1, angle_rightshoulder1, angle_leftshoulder1, angle_righthip1, angle_lefthip1,img1 = detect(cap1, 1)
    if angle_leftknee != 0 and angle_leftknee1 != 0:
        if angle_leftknee > angle_leftknee1+10:
            print("右膝盖角度小一些")
            #将文字写入图片中
            cv2.putText(img, "Right knee close a bit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 图片，文字内容， 坐标，字体，大小，颜色，字体厚度
        elif angle_leftknee < angle_leftknee1-10:
            print("右膝盖角度大一些")
            cv2.putText(img, "Right knee open a bit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if angle_rightknee != 0 and angle_rightknee1 != 0:
        if angle_rightknee > angle_rightknee1+10:
            print("左膝盖角度小一些")
            cv2.putText(img, "Left knee close a bit", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif angle_rightknee < angle_rightknee1-10:
            print("左膝盖角度大一些")
            cv2.putText(img, "Left knee open a bit", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if angle_rightelbow != 0 and angle_rightelbow1 != 0:
        if angle_rightelbow > angle_rightelbow1+10:
            print("左手肘角度小一些")
            cv2.putText(img, "Left elbow close a bit", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif angle_rightelbow < angle_rightelbow1-10:
            print("左手肘角度大一些")
            cv2.putText(img, "Left elbow open a bit", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if angle_leftelbow != 0 and angle_leftelbow1 != 0:
        if angle_leftelbow > angle_leftelbow1+10:
            print("右手肘角度小一些")
            cv2.putText(img, "Right elbow close a bit", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif angle_leftelbow < angle_leftelbow1-10:
            print("右手肘角度大一些")
            cv2.putText(img, "Right elbow open a bit", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if angle_rightshoulder != 0 and angle_rightshoulder1 != 0:
        if angle_rightshoulder > angle_rightshoulder1+10:
            print("左肩膀角度小一些")
            cv2.putText(img, "Left shoulder close a bit", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif angle_rightshoulder < angle_rightshoulder1-10:
            print("左肩膀角度大一些")
            cv2.putText(img, "Left shoulder open a bit", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if angle_leftshoulder != 0 and angle_leftshoulder1 != 0:
        if angle_leftshoulder > angle_leftshoulder1+10:
            print("右肩膀角度小一些")
            cv2.putText(img, "Right shoulder close a bit", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif angle_leftshoulder < angle_leftshoulder1-10:
            print("右肩膀角度大一些")
            cv2.putText(img, "Right shoulder open a bit", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if angle_righthip != 0 and angle_righthip1 != 0:
        if angle_righthip > angle_righthip1+10:
            print("左腿抬高一些")
            cv2.putText(img, "Left leg rise a bit", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif angle_righthip < angle_righthip1-10:
            print("左腿放低一些")
            cv2.putText(img, "Left leg down a bit", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if angle_lefthip != 0 and angle_lefthip1 != 0:
        if angle_lefthip > angle_lefthip1+10:
            print("右腿抬高一些")
            cv2.putText(img, "Right leg rise a bit", (50, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif angle_lefthip < angle_lefthip1-10:
            print("右腿放低一些")
            cv2.putText(img, "Right leg down a bit", (50, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    #镜像img和img1
    # img = cv2.flip(img, 1)
    # img1 = cv2.flip(img1, 1)
    cv2.imshow("img0", img)
    cv2.imshow("img1", img1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap0.release()
cv2.destroyAllWindows()
