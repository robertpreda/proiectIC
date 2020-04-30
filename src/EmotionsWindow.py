from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5 import QtGui

import os
import cv2


class EmotionsWindow(QMainWindow):
    def __init__(self, parent=None):
        super(EmotionsWindow, self).__init__(parent)
        self.resize(800, 600)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.toolbar = QToolBar("Camera2")  # Toolbar Widget
        self.toolbar.setIconSize(QSize(25, 20))
        self.addToolBar(self.toolbar)

        self.image_frame = QLabel()
        self.current_picture_number = 0
        self.current_folder_size = 0
        self.current_emotion = "None"

        photo_action = QAction(QIcon(os.path.join('../Resources/', 'happy-icon.png')), "Happy...", self)  # Display happy emotions
        photo_action.setStatusTip("Happy emotion")
        photo_action.triggered.connect(self.show_happy)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(os.path.join('../Resources/', 'sad-icon.png')), "Sad...", self)  # Display happy emotions
        photo_action.setStatusTip("Sad emotion")
        photo_action.triggered.connect(self.show_sad)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(os.path.join('../Resources/', 'disgust-icon.png')), "Disgust...", self)  # Display happy emotions
        photo_action.setStatusTip("Disgust emotion")
        photo_action.triggered.connect(self.show_disgust)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(os.path.join('../Resources/', 'fear-icon.png')), "Fear...", self)  # Display happy emotions
        photo_action.setStatusTip("Fear emotion")
        photo_action.triggered.connect(self.show_fear)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(os.path.join('../Resources/', 'left-arrow.png')), "Previous image...", self)  # See emotions example
        photo_action.setStatusTip("Previous")
        photo_action.triggered.connect(self.previous_image)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(os.path.join('../Resources/', 'right-arrow.png')), "Next image...",   self)  # See emotions example
        photo_action.setStatusTip("Next")
        photo_action.triggered.connect(self.next_image)
        self.toolbar.addAction(photo_action)

    def previous_image(self):
        if self.current_emotion != "None":
            self.current_picture_number -= 1
            if self.current_picture_number < 0:
                self.current_picture_number = self.current_folder_size-1

            self.show_emotions(self.current_emotion)

    def next_image(self):
        if self.current_emotion != "None":
            self.current_picture_number += 1
            if self.current_picture_number > self.current_folder_size-1:
                self.current_picture_number = 0

            self.show_emotions(self.current_emotion)

    def show_emotions(self, emotion):
        imagesList = os.listdir(f"../Resources/{emotion}/")

        cv_img = cv2.imread(f"../Resources/{emotion}/{imagesList[self.current_picture_number]}")
        img_resized = cv2.resize(cv_img, (800, 600))

        self.image = QtGui.QImage(img_resized.data, img_resized.shape[1], img_resized.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image_frame.show()
        self.setCentralWidget(self.image_frame)

    def show_happy(self):
        if self.current_emotion != "Happy":
            self.current_picture_number = 0
            self.current_emotion = "Happy"
            imagesList = os.listdir("../Resources/Happy/")
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_sad(self):
        if self.current_emotion != "Sad":
            self.current_picture_number = 0
            self.current_emotion = "Sad"
            imagesList = os.listdir("../Resources/Sad/")
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_disgust(self):
        if self.current_emotion != "Disgust":
            self.current_picture_number = 0
            self.current_emotion = "Disgust"
            imagesList = os.listdir("../Resources/Disgust/")
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_fear(self):
        if self.current_emotion != "Fear":
            self.current_picture_number = 0
            self.current_emotion = "Fear"
            imagesList = os.listdir("../Resources/Fear/")
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return