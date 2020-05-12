from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QToolBar, QLabel, QAction, QWidget, QSizePolicy
from PyQt5.QtCore import QSize

from src.utils import resource_path
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

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/neutral-icon.png')), "Neutral...", self)  # Display happy emotions
        photo_action.setStatusTip("Neutral emotion")
        photo_action.triggered.connect(self.show_neutral)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/happy-icon.png')), "Happy...", self)  # Display happy emotions
        photo_action.setStatusTip("Happy emotion")
        photo_action.triggered.connect(self.show_happy)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/surprise-icon.png')), "Surprise...", self)  # Display happy emotions
        photo_action.setStatusTip("Surprise emotion")
        photo_action.triggered.connect(self.show_surprise)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/sad-icon.png')), "Sad...", self)  # Display happy emotions
        photo_action.setStatusTip("Sad emotion")
        photo_action.triggered.connect(self.show_sad)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/disgust-icon.png')), "Disgust...", self)  # Display happy emotions
        photo_action.setStatusTip("Disgust emotion")
        photo_action.triggered.connect(self.show_disgust)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/fear-icon.png')), "Fear...", self)  # Display happy emotions
        photo_action.setStatusTip("Fear emotion")
        photo_action.triggered.connect(self.show_fear)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/angry-icon.png')), "Angry...", self)  # Display happy emotions
        photo_action.setStatusTip("Angry emotion")
        photo_action.triggered.connect(self.show_angry)
        self.toolbar.addAction(photo_action)

        self.spacer = QWidget()
        self.spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.addWidget(self.spacer)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/left-arrow-icon.png')), "Previous image...", self)  # See emotions example
        photo_action.setStatusTip("Previous")
        photo_action.triggered.connect(self.previous_image)
        self.toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/right-arrow-icon.png')), "Next image...",   self)  # See emotions example
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
        imagesList = os.listdir(resource_path(f"../Resources/{emotion}/"))

        cv_img = cv2.imread(resource_path(f"../Resources/{emotion}/{imagesList[self.current_picture_number]}"))
        img_resized = cv2.resize(cv_img, (800, 600))

        self.image = QImage(img_resized.data, img_resized.shape[1], img_resized.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(self.image))

        self.image_frame.show()
        self.setCentralWidget(self.image_frame)

    def show_neutral(self):
        if self.current_emotion != "Neutral":
            self.current_picture_number = 0
            self.current_emotion = "Neutral"
            imagesList = os.listdir(resource_path("../Resources/Neutral/"))
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_happy(self):
        if self.current_emotion != "Happy":
            self.current_picture_number = 0
            self.current_emotion = "Happy"
            imagesList = os.listdir(resource_path("../Resources/Happy/"))
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_surprise(self):
        if self.current_emotion != "Surprise":
            self.current_picture_number = 0
            self.current_emotion = "Surprise"
            imagesList = os.listdir(resource_path("../Resources/Surprise/"))
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_sad(self):
        if self.current_emotion != "Sad":
            self.current_picture_number = 0
            self.current_emotion = "Sad"
            imagesList = os.listdir(resource_path("../Resources/Sad/"))
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_disgust(self):
        if self.current_emotion != "Disgust":
            self.current_picture_number = 0
            self.current_emotion = "Disgust"
            imagesList = os.listdir(resource_path("../Resources/Disgust/"))
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_fear(self):
        if self.current_emotion != "Fear":
            self.current_picture_number = 0
            self.current_emotion = "Fear"
            imagesList = os.listdir(resource_path("../Resources/Fear/"))
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return

    def show_angry(self):
        if self.current_emotion != "Angry":
            self.current_picture_number = 0
            self.current_emotion = "Angry"
            imagesList = os.listdir(resource_path("../Resources/Angry/"))
            self.current_folder_size = len(imagesList)

        self.show_emotions(self.current_emotion)
        return