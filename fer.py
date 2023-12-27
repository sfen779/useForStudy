#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QPainter, QBrush, QColor, QPen
import os
import time
#import io
#import base64
#import json
import time
import fastdeploy as fd
import argparse
#import paddle.inference as paddle_infer
#from log import logger


def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    normalized_data = (nparray - np.min(nparray)) / (np.max(nparray) - np.min(nparray))
    return normalized_data


def check_feat(feature1, feature2):
    """Check the feature similarity.
    用于计算两个特征的相似度，返回值越小？越相似。
    是之前的reid项目历史遗留
    """
    s1 = np.dot(feature1,feature1.T)
    s2 = np.dot(feature2,feature2.T)
    s12 = np.dot(feature1,feature2.T)
    dist = (1 - s12/np.sqrt(s1*s2)) / 2
    dist = (dist - 0.5) * 10 + 1

    return dist


def preprocess(image_path, image_height, image_width):
    """Preprocess the image for inference.
    预处理图片，用于模型推理
    主要是resize和transpose
    """
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img



def parse_arguments():
    """Parse the command line arguments.
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="My Qt Application")
    parser.add_argument("--param", default=0 ,type=str, help="A parameter for the Qt application")
    return parser.parse_args()



class mps_card_lahen_segment:

    __tag = -1

    def __init__(self, tag):

        self.__tag = tag
        # model_file = "people_det.onnx"
        model_file = "posterV2_7cls.onnx"
        # params_file = "inference_model/model.pdiparams"
        # config_file = "inference_model/model.yaml"
        runtime_option = fd.RuntimeOption()
        # imported fastdeploy as fd
        runtime_option.use_ort_backend()
        runtime_option.use_gpu()
        self.model = fd.vision.detection.YOLOv5(
            model_file, runtime_option=runtime_option
        )
        self.save_dir = 'images'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def classify_by_image(self, image):

        # print('in mps_sample_by_image')
        # print(self.__tag)
        #################### write your function here ####################
        img = cv2.imread(image)
        result = self.model.predict(img, conf_threshold=0.5, nms_iou_threshold=0.5)

        if result.boxes == []:
            msg = 200
        else:
            box = []
            for n, i in enumerate(result.label_ids):
                if i == 0:
                    box.append(result.boxes[n])
            msg = box
        return msg
#mps是干什么用的？--new一个刚刚定义的类，mps_card_lahen_segment
#mps代表公司名-micro pattern software
mps = mps_card_lahen_segment(-1)



class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        # self.setBac
        # self.face_recong = face.Recognition()
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        # self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.number = 0
        self.sim = 0
        arg = parse_arguments()

        self.arg = arg.param

    def set_ui(self):
        """初始化程序界面"""
        font = QtGui.QFont()
        font.setFamily("kaiti")
        font.setPointSize(18)
        self.textBrowser = QtWidgets.QLabel("人体检测app")
        self.textBrowser.setAlignment(Qt.AlignCenter)
        self.textBrowser.setFont(font)

        # self.label.setText(_translate("MainWindow", "TextLabel"))
        self.mm_layout = QVBoxLayout()
        self.l_down_widget = QtWidgets.QWidget()
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        self.button_open_camera = QtWidgets.QPushButton(u'开始检测')
        # self.button_cap = QtWidgets.QPushButton(u'拍照')
        #
        # self.canshu = QtWidgets.QPushButton(u'参数设置')
        # self.det = QtWidgets.QPushButton(u'人体检测')
        fontx = QtGui.QFont()
        fontx.setFamily("kaiti")
        fontx.setPointSize(16)

        # Button 的颜色修改
        button_color = [self.button_open_camera]
        for i in range(1):
            button_color[i].setFont(fontx)
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{background-color:rgb(78,255,255)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:10px}"
                                          "QPushButton{padding:2px 4px}")

        self.button_open_camera.setMinimumHeight(50)
       # self.button_cap.setMinimumHeight(50)
        # self.canshu.setMinimumHeight(50)
#        self.det.setMinimumHeight(50)

        # move()方法移动窗口在屏幕上的位置到x = 300，y = 300坐标。
        self.move(500, 500)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(100, 100)

        self.label_show_camera.setFixedSize(1921, 1081)
        # self.label_show_camera.setFixedSize(1300, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
#        self.__layout_fun_button.addWidget(self.button_cap)
        # self.__layout_fun_button.addWidget(self.canshu)
#        self.__layout_fun_button.addWidget(self.det)
        self.__layout_fun_button.addWidget(self.label_move)
        # 添加一个右侧的组件
        self.right_widget = QWidget()
        self.right_widget_layout = QHBoxLayout()
        self.cap_label = QLabel()
        self.cap_label.setFixedSize(1921, 1081)
        # self.label_show_camera.setFixedSize(1300, 481)
        self.cap_label.setAutoFillBackground(False)
        self.right_widget_layout.addWidget(self.label_show_camera)
        #self.right_widget_layout.addWidget(self.cap_label)
        self.right_widget.setLayout(self.right_widget_layout)

        self.__layout_main.addWidget(self.right_widget)
        self.__layout_main.addLayout(self.__layout_fun_button)
        # self.__layout_main.addWidget(self.label_show_camera)


        # self.setLayout(self.__layout_main)
        self.l_down_widget.setLayout(self.__layout_main)
        self.mm_layout.addWidget(self.textBrowser)
        self.mm_layout.addWidget(self.l_down_widget)
        self.setLayout(self.mm_layout)
        self.label_move.raise_()
        self.setWindowTitle(u'人体检测app')
        # self.setStyleSheet("#MainWindow{border-image:url(DD.png)}")

        '''
        # 设置背景图片
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('background.jpg')))
        self.setPalette(palette1)
        '''

    def slot_init(self):
        """初始化槽函数"""
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
#        self.button_cap.clicked.connect(self.capx)
#        self.det.clicked.connect(self.button_det_people)

    def button_det_people(self):
        """使用QPaintr绘制一个矩形框"""
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setBrush(QBrush(QColor(255, 0, 0)))  # 设置矩形填充颜色为红色
        qp.setPen(QPen(QColor(0, 0, 0)))  # 设置矩形边框颜色为黑色
        qp.drawRect(50, 50, 200, 200)  # 在 (50, 50) 位置绘制一个 200x200 大小的矩形框
        qp.end()

    def button_open_camera_click(self):
        """Slot function to open/close camera."""   
        #槽函数打开/关闭相机。
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM,  cv2.CAP_DSHOW)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)

                self.button_open_camera.setText(u'开始检测')
                # while flag:
                #     qp = QPainter()
                #     qp.begin(self)
                #     qp.setRenderHint(QPainter.Antialiasing)
                #     qp.setBrush(QBrush(QColor(255, 0, 0)))  # 设置矩形填充颜色为红色
                #     qp.setPen(QPen(QColor(0, 0, 0)))  # 设置矩形边框颜色为黑色
                #     qp.drawRect(50, 50, 200, 200)  # 在 (50, 50) 位置绘制一个 200x200 大小的矩形框
                #     qp.end()
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'打开相机')


    def show_camera(self):
        """Slot function to capture frame and show in label."""
        #槽函数捕获帧并在标签中显示。
        self.number += 1
        flag, self.image = self.cap.read()
        # face = self.face_detect.align(self.image)
        # if face:
        #     pass
        show = cv2.resize(self.image, (1920, 960))

        image = cv2.imwrite('test.jpg', show)

        path = 'test.jpg'
        msg = mps.classify_by_image(path)

        if msg == 200:
            show = self.image
        else:
            if len(msg) == 1:
                    cv2.rectangle(show, (int(msg[0][0]), int(msg[0][1])), (int(msg[0][2]), int(msg[0][3])), (0, 0, 255),3)
                    text = '{} targets in the picture'.format(1)  ##编辑文本
                    fontScale = 2 # 字体缩放比例
                    color = (0, 0, 255)  # 字体颜色
                    pos = (700, 40)  # 位置
                    cv2.putText(show, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 3)
                    if int(self.arg) < 1:
                        text2 = 'warning break-in!'  ##编辑文本
                    else:
                        text2 = 'leaving!'  ##编辑文本
                    fontScale = 2  # 字体缩放比例
                    color = (255, 0, 0)  # 字体颜色
                    pos = (700, 100)  # 位置
                    cv2.putText(show, text2, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 3)                    
            else:
                for i in range(len(msg)):
                    cv2.rectangle(show, (int(msg[i][0]), int(msg[i][1])), (int(msg[i][2]), int(msg[i][3])),(0, 0, 255), 3)
                    text = '{} targets in the picture'.format(len(msg))  ##编辑文本
                    fontScale = 2  # 字体缩放比例
                    color = (0, 0, 255)  # 字体颜色
                    pos = (700, 40)  # 位置
                    cv2.putText(show, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 3)        
                        
                if int(self.arg) < len(msg):
                    text = 'warning break-in!'.format(len(msg))  ##编辑文本
                else:
                    text = 'leaving!'.format(len(msg))
                fontScale = 2  # 字体缩放比例
                color = (255, 0, 0)  # 字体颜色
                pos = (700, 100)  # 位置
                cv2.putText(show, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 3)

        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        # print(show.shape[1], show.shape[0])
        # show.shape[1] = 640, show.shape[0] = 480
        self.showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
        # self.x += 1
        # self.label_move.move(self.x,100)

        # if self.x ==320:
        #     self.label_show_camera.raise_()


    def capx(self):
        """Slot function to capture frame and save in local."""
        #槽函数捕获帧并在本地保存。
        FName = fr"images\cap{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        # cv2.imwrite(FName + ".jpg", self.image)
        # self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.cap_label.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
        self.showImage.save(FName + ".jpg", "JPG", 100)


    def closeEvent(self, event):
        """Close the main window."""
        #关闭主窗口。
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()





if __name__ == "__main__":

    App = QApplication(sys.argv)
    ex = Ui_MainWindow()
    # ex.setStyleSheet("#MainWindow{border-image:url(DD.png)}")
    ex.show()
    sys.exit(App.exec_())