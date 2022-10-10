#20220911更新日志：在识别后准星移动前进行img_val的截取，用于比对下一循环获取的图片是否和之前重复导致连续移动鼠标两次
import math
import sys
import time
import cv2
import torch
import win32api
import win32con
import win32gui
import pyautogui
from PyQt5.QtWidgets import QApplication
from pynput.mouse import Controller
import mouse
import mss
import numpy as np
import dxcam
from PIL import Image
import imageio

f = open('log.txt','w')

# 这里这俩class就是文章上面说的那个传入两个坐标点，计算直线距离的
class Point():
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class Line(Point):
    def __init__(self, x1, y1, x2, y2):
        super().__init__(x1, y1, x2, y2)

    def getlen(self):
        changdu = math.sqrt(math.pow((self.x1 - self.x2), 2) + math.pow((self.y1 - self.y2), 2))
        return changdu


# 第一步：我们获取到某FPS游戏的窗口句柄
hwnd = win32gui.FindWindow(None, "东南亚-JZHyyds")


# 这个方法是获取上面句柄窗口的窗口截图，用的是PyQt截图，有速度更快更好的方式的话可以换上
# 截图完毕后保存在根目录的cfbg.bmp文件

camera = dxcam.create()
frame = camera.grab()
camera.start(target_fps=240)

# 这里就是调用我们那yolo模型来进行推理啦，我设置的是cuda，也就是英伟达的GPU，因为cpu太慢了。
# 如果自己的不能使用GPU推理的话把下面这两行改改，改成cpu的就可以了。
device = torch.device("cuda")
model = torch.hub.load('F:/zgcyxzym9/feeeeee/yolov5-master', 'custom', 'F:/zgcyxzym9/feeeeee/yolov5-master/runs/train/exp11/weights/best.pt',
                       source='local', force_reload=False)  # 加载本地模型
model = model.to(device)
# 这里是定义屏幕宽高[其实这俩就是游戏所对应的分辨率，比如：游戏里1920*1080这里就是1920*1080]
game_width = 1920
game_height = 1080
# 这边就是开始实时进行游戏窗口推理了
# 无限循环 -> 截取屏幕 -> 推理模型获取到每个敌人坐标 -> 计算每个敌人中心坐标 -> 挑选距离准星最近的敌人 -> 如果左键是按下状态则控制鼠标移动到敌人的身体或者头部(本文计算方式是移动到头部)
monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
lastCD = 0
#以下是调试参数记得删除
last_status = 0
prev_status = 0


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def get_multiplier(dist):
    if(dist<20):
        ans = dist*0.01 + 0.8
    elif(dist<100):
        if(dist>70):
            ans = 1.6
        else:
            ans = (dist-20)*0.01 + 1.1
    else:
        ans = (dist-100)*0.001 + 1.6
    return 1

def get_multiplier_alt(dist):
    return -3.43753667257454+2.36385928061026*dist+0.000992319674743902*dist*dist

img_val = camera.get_latest_frame()

while True:
    # 截取屏幕
    img = camera.get_latest_frame()
    # 比对验证图片
    difference = cv2.subtract(img, img_val)
    result = not np.any(difference)
    if result is True:
        print("con")
        continue
    # 开始推理
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
        if classitem == 1 and conf > 0.6:
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
            mouseModal = Controller()
            x, y = mouseModal.position
            L1 = Line(x, y, xindex, yindex)
            # 获取到距离并且存放在cdList集合中
            cdList.append(int(L1.getlen()))
            xyList.append([xindex, yindex, listItem[0], listItem[1], listItem[2], listItem[3]])
        # 这里就得到了距离最近的敌人位置了
        minCD = min(cdList)
        if(minCD == lastCD):
            continue
        # 如果认为这一张图片值得进行训练则保存下来
        if(last_status == 1 and prev_status == 0):
            print("possible model failure\n")
            save_route = "C:/Users/hhdsj/Desktop/temp/"+str(time.time())
            # cv2.imwrite(save_route,img)
            imageio.imwrite(save_route+".jpg", img) # 20220919修改 使用新代码使输出为RGB
            # 20220919增加 自动生成标签
            fd = open(save_route+".txt", 'w+')
            for xyItem in xyList:
                fd.write("%d %f %f %f %f\n" % (1, (xyItem[2]+(xyItem[4]-xyItem[2])/2)/game_width, (xyItem[3]+(xyItem[5]-xyItem[3])/2)/game_height, (xyItem[4]-xyItem[2])/game_width, (xyItem[5]-xyItem[3])/game_height))
            fd.close()

        last_status=prev_status
        prev_status=1

        # multiplier为一个玄学系数，距离越远越大，防止准星晃动的同时加速瞄准
        if(minCD>0):
            multiplier = get_multiplier_alt(minCD)/minCD
        else:
            multiplier = 1
        if(multiplier<1):
            multiplier=1
        # 获取比对图片
        img_val = camera.get_latest_frame()
        # 如果敌人距离鼠标坐标小于150则自动进行瞄准，这里可以改大改小，小的话跟枪会显得自然些
        if minCD < 300:
            for cdItem, xyItem in zip(cdList, xyList):
                if cdItem == minCD:
                    # 锁头算法：使用win32api获取左键按下状态，如果按下则开始自动跟枪
                    #if win32api.GetAsyncKeyState(0x01):
                    if not(win32api.GetAsyncKeyState(ord('V'))):
                        # 控制鼠标移动到某个点：看不懂计算方式的话看文章下面讲解吧O(∩_∩)O
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(multiplier*(xyItem[0] - game_width // 2)),
                                             int(multiplier*(xyItem[1] - game_height // 2)), 0, 0)
                        lastCD = minCD
                        print(minCD)
                    else:
                        lastCD = -1
                    break
    #调试
    else:
        last_status = prev_status
        prev_status=0
