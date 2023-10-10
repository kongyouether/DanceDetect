import json
import numpy as np
import cv2
import math

def drawpoint(index,frame):
    # 读取json文件,文件名格式为data+index+.txt，其中index为0,1,2,3,4
    with open('data' + str(index) + '.txt', 'r') as file:
        data = json.load(file)  # 解析 JSON 数据

        # 读取图片
        img = frame
        hh= img.shape[0]
        ww= img.shape[1]
        # print("ww,hh", ww, hh)
        # 提取 confidence 值
        confidence = data[0]['confidence']
        # 如果提取不到confidence的值，就return
        if confidence == 0:
            return 0 # 0表示没有检测到人体

        x1 = data[0]['box']['x1']
        y1 = data[0]['box']['y1']
        x2 = data[0]['box']['x2']
        y2 = data[0]['box']['y2']
        # print(confidence)
        # print("x1,y1,x2,y2", x1, y1, x2, y2)
        x1 = int(x1 * ww)
        y1 = int(y1 * hh)
        x2 = int(x2 * ww)
        y2 = int(y2 * hh)
        # print("x1,y1,x2,y2", x1, y1, x2, y2)
        boxwidth = x2 - x1
        boxheight = y2 - y1
        # 画出框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 参数：图片、左上角坐标、右下角坐标、颜色、线宽

        # 提取 keypoints 值
        keypointsX = data[0]['keypoints']['x']
        keypointsY = data[0]['keypoints']['y']
        keypointsV = data[0]['keypoints']['visible']
        # print("keypointsX", keypointsX)
        # print("keypointsY", keypointsY)

        # 0:鼻子 1:左眼 2:右眼 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 9:左腕 10:右腕 11:左髋 12:右髋 13:左膝 14:右膝 15:左踝 16:右踝
        # 计算肢体之间的角度
        # 左肩，左髋，左膝，左踝
        leftshoulder = (int(keypointsX[5] * ww), int(keypointsY[5] * hh))
        lefthip = (int(keypointsX[11] * ww), int(keypointsY[11] * hh))
        leftknee = (int(keypointsX[13] * ww), int(keypointsY[13] * hh))
        leftankle = (int(keypointsX[15] * ww), int(keypointsY[15] * hh))
        # 右肩，右髋，右膝，右踝
        rightshoulder = (int(keypointsX[6] * ww), int(keypointsY[6] * hh))
        righthip = (int(keypointsX[12] * ww), int(keypointsY[12] * hh))
        rightknee = (int(keypointsX[14] * ww), int(keypointsY[14] * hh))
        rightankle = (int(keypointsX[16] * ww), int(keypointsY[16] * hh))
        # 左肘，左腕
        leftelbow = (int(keypointsX[7] * ww), int(keypointsY[7] * hh))
        leftwrist = (int(keypointsX[9] * ww), int(keypointsY[9] * hh))
        # 右肘，右腕
        rightelbow = (int(keypointsX[8] * ww), int(keypointsY[8] * hh))
        rightwrist = (int(keypointsX[10] * ww), int(keypointsY[10] * hh))
        # 左眼，左耳
        lefteye = (int(keypointsX[1] * ww), int(keypointsY[1] * hh))
        leftear = (int(keypointsX[3] * ww), int(keypointsY[3] * hh))
        # 右眼，右耳
        righteye = (int(keypointsX[2] * ww), int(keypointsY[2] * hh))
        rightear = (int(keypointsX[4] * ww), int(keypointsY[4] * hh))
        # 鼻子
        nose = (int(keypointsX[0] * ww), int(keypointsY[0] * hh))

        #计算肢体之间的角度
        # 把angle_rightshoulder,angle_leftshoulder,angle_rightelbow,angle_leftelbow,angle_righthip,angle_lefthip,angle_rightknee,angle_leftknee 画在图片上
        # 利用余弦定理计算rightshoulder,rightelbow,righthip形成的三角形中rightshoulder顶点的角度
        angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip = 0, 0, 0, 0, 0, 0, 0, 0

        if keypointsV[6] > 0.5:
            angle_rightshoulder = calculate_angle(rightshoulder, rightelbow, righthip)+0.01
            cv2.putText(img, str(int(angle_rightshoulder)), rightshoulder, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 利用余弦定理计算leftshoulder,leftelbow,lefthip形成的三角形中leftshoulder顶点的角度
        if keypointsV[5] > 0.5:
            angle_leftshoulder = calculate_angle(leftshoulder, leftelbow, lefthip)+0.01
            cv2.putText(img, str(int(angle_leftshoulder)), leftshoulder, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 利用余弦定理计算rightelbow,rightshoulder,rightwrist形成的三角形中rightelbow顶点的角度
        if keypointsV[8] > 0.5:
            angle_rightelbow = calculate_angle(rightelbow, rightshoulder, rightwrist)+0.01
            cv2.putText(img, str(int(angle_rightelbow)), rightelbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 利用余弦定理计算leftelbow,leftshoulder,leftwrist形成的三角形中leftelbow顶点的角度
        if keypointsV[7] > 0.5:
            angle_leftelbow = calculate_angle(leftelbow, leftshoulder, leftwrist)+0.01
            cv2.putText(img, str(int(angle_leftelbow)), leftelbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 利用余弦定理计算righthip,rightknee,rightshoulder形成的三角形中righthip顶点的角度
        if keypointsV[12] > 0.5:
            angle_righthip = calculate_angle(righthip, rightknee, rightshoulder)+0.01
            cv2.putText(img, str(int(angle_righthip)), righthip, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 利用余弦定理计算lefthip,leftknee,leftshoulder形成的三角形中lefthip顶点的角度
        if keypointsV[11] > 0.5:
            angle_lefthip = calculate_angle(lefthip, leftknee, leftshoulder)+0.01
            cv2.putText(img, str(int(angle_lefthip)), lefthip, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 利用余弦定理计算rightknee,righthip,rightankle形成的三角形中rightknee顶点的角度
        if keypointsV[14] > 0.5:
            angle_rightknee = calculate_angle(rightknee, righthip, rightankle)+0.01
            cv2.putText(img, str(int(angle_rightknee)), rightknee, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 利用余弦定理计算leftknee,lefthip,leftankle形成的三角形中leftknee顶点的角度
        if keypointsV[13] > 0.5:
            angle_leftknee = calculate_angle(leftknee, lefthip, leftankle)+0.01
            cv2.putText(img, str(int(angle_leftknee)), leftknee, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #创建100*100的图片

        copyimg = img.copy()
        #在图片上画出keypoints
        for i in range(0, len(keypointsX)):
            realx = int(keypointsX[i]*ww)
            realy = int(keypointsY[i]*hh)
            # print("realx,realy", realx, realy)
            if keypointsV[i] > 0.5:
                cv2.circle(copyimg, (realx, realy), 5, (0, 0, 255), 5) # 参数：图片、圆心、半径、颜色、线宽

        # 在图片上画出肢体,只有当肢体的visible大于0.5时才画出
        # 左肩-左髋
        if keypointsV[5] > 0.5 and keypointsV[11] > 0.5:
            cv2.line(copyimg, leftshoulder, lefthip, (0, 255, 0), 2)
        # 右肩-右髋
        if keypointsV[6] > 0.5 and keypointsV[12] > 0.5:
            cv2.line(copyimg, rightshoulder, righthip, (0, 255, 0), 2)
        # 左肩-右肩
        if keypointsV[5] > 0.5 and keypointsV[6] > 0.5:
            cv2.line(copyimg, leftshoulder, rightshoulder, (0, 255, 0), 2)
        # 左髋-右髋
        if keypointsV[11] > 0.5 and keypointsV[12] > 0.5:
            cv2.line(copyimg, lefthip, righthip, (0, 255, 0), 2)
        # 左肩-左肘
        if keypointsV[5] > 0.5 and keypointsV[7] > 0.5:
            cv2.line(copyimg, leftshoulder, leftelbow, (0, 255, 0), 2)
        # 右肩-右肘
        if keypointsV[6] > 0.5 and keypointsV[8] > 0.5:
            cv2.line(copyimg, rightshoulder, rightelbow, (0, 255, 0), 2)
        # 左肘-左腕
        if keypointsV[7] > 0.5 and keypointsV[9] > 0.5:
            cv2.line(copyimg, leftelbow, leftwrist, (0, 255, 0), 2)
        # 右肘-右腕
        if keypointsV[8] > 0.5 and keypointsV[10] > 0.5:
            cv2.line(copyimg, rightelbow, rightwrist, (0, 255, 0), 2)
        # 左髋-左膝
        if keypointsV[11] > 0.5 and keypointsV[13] > 0.5:
            cv2.line(copyimg, lefthip, leftknee, (0, 255, 0), 2)
        # 右髋-右膝
        if keypointsV[12] > 0.5 and keypointsV[14] > 0.5:
            cv2.line(copyimg, righthip, rightknee, (0, 255, 0), 2)
        # 左膝-左踝
        if keypointsV[13] > 0.5 and keypointsV[15] > 0.5:
            cv2.line(copyimg, leftknee, leftankle, (0, 255, 0), 2)
        # 右膝-右踝
        if keypointsV[14] > 0.5 and keypointsV[16] > 0.5:
            cv2.line(copyimg, rightknee, rightankle, (0, 255, 0), 2)

        # cv2.imshow("img"+ str(index), copyimg)
        # print("angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip", angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip)
        return angle_leftknee, angle_rightknee, angle_rightelbow, angle_leftelbow, angle_rightshoulder, angle_leftshoulder, angle_righthip, angle_lefthip,copyimg
        # cv2.waitKey(0)

def calculate_angle(Pointa, Pointb, Pointc):
    # 利用余弦定理计算a,b,c形成的三角形中a顶点的角度
    a = math.sqrt(math.pow(Pointa[0] - Pointb[0], 2) + math.pow(Pointa[1] - Pointb[1], 2)) + 0.01
    b = math.sqrt(math.pow(Pointa[0] - Pointc[0], 2) + math.pow(Pointa[1] - Pointc[1], 2)) + 0.01
    c = math.sqrt(math.pow(Pointb[0] - Pointc[0], 2) + math.pow(Pointb[1] - Pointc[1], 2)) + 0.01
    # print("a,b,c:", a, b, c)
    angle = math.acos((a * a + b * b - c * c) / (2 * a * b)) * 180 / math.pi
    return angle
if __name__ == '__main__':
    drawpoint(0)
