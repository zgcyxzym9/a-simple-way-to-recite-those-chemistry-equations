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

camera = dxcam.create()
frame = camera.grab()
camera.start(target_fps=240)
device = torch.device("cuda")
model = torch.hub.load('F:/zgcyxzym9/feeeeee/cg_new_full', 'custom', 'F:/zgcyxzym9/feeeeee/cg_new_full/runs/train/20221228gen8/weights/best.pt',
                       source='local', force_reload=False)  # 加载本地模型
model = model.to(device)
game_width = 1024
game_height = 768

def GetMultiplier(MoveDist):
    return 1

def CalcCore():
    global img
    global game_height
    global game_width
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
        if (classitem == 1 or classitem == 0) and conf > 0.5:
            newlist.append([int(xmin), int(ymin), int(xmax), int(ymax), conf])
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
        multiplier = GetMultiplier(minCD)
        if minCD < 300:
            for cdItem, xyItem in zip(cdList, xyList):
                if cdItem == minCD:
                    if not(win32api.GetAsyncKeyState(ord('V'))):
                        CrossHairX = game_width // 2
                        CrossHairY = game_height // 2
                        if(CrossHairX < xyItem[4] and CrossHairX > xyItem[2] and CrossHairY < xyItem[5] and CrossHairY > xyItem[3]):   #应该可以说明已经瞄准了
                            if(not (win32api.GetAsyncKeyState(1))):
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(multiplier*(xyItem[0] - game_width // 2)),
                                             int(multiplier*(xyItem[1] - game_height // 2)), 0, 0)
        return 1

    for xmin, ymin, xmax, ymax, classitem, conf in zip(xmins, ymins, xmaxs, ymaxs, class_list, confidences):
        if (classitem == 1 or classitem == 0) and conf > 0.4:
            save_route = "C:/Users/hhdsj/Desktop/temp/"+str(time.time())
            imageio.imwrite(save_route+".jpg", img) # 20220919修改 使用新代码使输出为RGB
            break

    return 0

def Main():
    global img
    global game_width
    global game_height
    LastRunWithResult = 0
    while True:     #程序主循环
        img = camera.get_latest_frame()     #先获取主图像
        if(LastRunWithResult):       #上次有结果了，缩小搜索范围
            img = img[225:768-225, 300:1024-300, :]
            game_width = 1024 - 600
            game_height = 768 - 450
        else:
            img = img[75:768-75, 100:1024-100, :]
            game_width = 1024 - 200
            game_height = 768 - 150
        LastRunWithResult = CalcCore()


































































Main()
