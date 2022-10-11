import cv2
import os
import numpy as np
import torch

def read_path(file_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        ####change to gray
      #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #####save figure
        cv2.imwrite('*/grayvoc/trainval/VOCdevkit/VOC2007/JPEGImages'+"/"+filename,image_np)

#注意*处如果包含家目录（home）不能写成~符号代替
#必须要写成"/home"的格式，否则会报错说找不到对应的目录
#读取的目录

device = torch.device("cuda")
model = torch.hub.load('F:/zgcyxzym9/feeeeee/yolov5-master', 'custom', 'F:/zgcyxzym9/feeeeee/yolov5-master/runs/train/exp11/weights/best.pt',
                       source='local', force_reload=False)  # 加载本地模型
model = model.to(device)
game_width = 1920
game_height = 1080

file_pathname = "C:/Users/hhdsj/Desktop/temp/read"
for filename in os.listdir(file_pathname):
    img = cv2.imread(file_pathname+'/'+filename)
    results = model(img)
    # 过滤模型
    xmins = results.pandas().xyxy[0]['xmin']
    ymins = results.pandas().xyxy[0]['ymin']
    xmaxs = results.pandas().xyxy[0]['xmax']
    ymaxs = results.pandas().xyxy[0]['ymax']
    class_list = results.pandas().xyxy[0]['class']
    confidences = results.pandas().xyxy[0]['confidence']
    newlist = []
    for xmin, ymin, xmax, ymax, classitem, conf in zip(xmins, ymins, xmaxs, ymaxs, class_list, confidences):
        if classitem == 1 and conf > 0.5:
            newlist.append([int(xmin), int(ymin), int(xmax), int(ymax), conf])
            # print(int(xmin), int(ymin), int(xmax), int(ymax), conf)  #输出坐标
    # 循环遍历每个敌人的坐标信息传入距离计算方法获取每个敌人距离鼠标的距离
    if len(newlist) > 0:
        # 存放距离数据
        cdList = []
        xyList = []
        for listItem in newlist:
            # 当前遍历的人物中心坐标
            xindex = int(listItem[2] - (listItem[2] - listItem[0]) / 2)
            yindex = int(listItem[3] - (listItem[3] - listItem[1]) / 2)
            xyList.append([xindex, yindex, listItem[0], listItem[1], listItem[2], listItem[3]])

        # 20220919增加 自动生成标签
        save_route = "C:/Users/hhdsj/Desktop/temp/zzz/lab"
        #save_name = filename - ".jpg"
        save_name = os.path.splitext(filename)[0]
        fd = open(save_route + "/" + save_name + ".txt", 'w+')
        for xyItem in xyList:
            fd.write("%d %f %f %f %f\n" % (1, (xyItem[2]+(xyItem[4]-xyItem[2])/2)/game_width, (xyItem[3]+(xyItem[5]-xyItem[3])/2)/game_height, (xyItem[4]-xyItem[2])/game_width, (xyItem[5]-xyItem[3])/game_height))
        fd.close() #20221010修改 在修改输出图片为未被识别的图片后标签使用新程序获得