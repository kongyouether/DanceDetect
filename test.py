import cv2

# 连续读取文件夹内cam1, cam2, cam3, cam4的图片
for i in range(1, 100):
    img1 = cv2.imread('cam1/' + str(i) + '.jpg')
    img2 = cv2.imread('cam2/' + str(i) + '.jpg')
    img3 = cv2.imread('cam3/' + str(i) + '.jpg')
    img4 = cv2.imread('cam4/' + str(i) + '.jpg')
    # 将四张图片合并为一张图片
    img = cv2.hconcat([img1, img2, img3, img4])
    # 显示图片
    cv2.imshow('img', img)
    # 每隔1ms刷新一次图片
    cv2.waitKey(1)