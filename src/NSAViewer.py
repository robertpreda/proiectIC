from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QToolBar, QAction, QComboBox, QFileDialog, QLabel, QWidget, QErrorMessage, QApplication
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtMultimedia import QCameraInfo, QCamera, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder

from torchvision import transforms
from src.EmotionsWindow import EmotionsWindow
from src.facial_landmarks import init_facial_landmarks_detector, init_emotions_detector, detect_landmarks
from src.utils import resource_path

import cv2
import sys
import time


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.resize(1024, 768)
        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            pass #quit

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.save_path = "../Snapshots/"  # Initial save path is in Snapshots folder
        self.save_seq = 0

        self.currently_shown = "none"  # Neither an image or video is shown at first
        # Setup tools
        camera_toolbar = QToolBar("Camera")  # Toolbar Widget
        camera_toolbar.setIconSize(QSize(25, 20))
        self.addToolBar(camera_toolbar)

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(3, 768)  # Set camera width
        self.camera.set(4, 1024)  # Set camera height

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/snapshot-icon.png')), "Take photo...", self)  # Snapshot button
        photo_action.setStatusTip("Snapshot!")
        photo_action.triggered.connect(self.take_photo)
        camera_toolbar.addAction(photo_action)

        change_folder_action = QAction(QIcon(resource_path('../Resources/Icons/folder-icon.png')), "Change save location...", self)  # Change snapshot save location
        change_folder_action.setStatusTip("Change the folder where snapshots are saved.")
        change_folder_action.triggered.connect(self.change_folder)
        camera_toolbar.addAction(change_folder_action)

        camera_selector = QComboBox()  # Choose between laptop cameras
        camera_selector.addItems([c.description() for c in self.available_cameras])
        camera_selector.currentIndexChanged.connect(self.select_camera)
        camera_toolbar.addWidget(camera_selector)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/load-image.png')), "Load image...", self)  # Load image from system
        photo_action.setStatusTip("Load image from system.")
        photo_action.triggered.connect(self.show_image)
        camera_toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/video-icon.png')), "Live video...", self)  # Start live video
        photo_action.setStatusTip("Live video recognition.")
        photo_action.triggered.connect(self.start_timer)
        camera_toolbar.addAction(photo_action)

        photo_action = QAction(QIcon(resource_path('../Resources/Icons/emotions-icon.jpg')), "Emotions example...", self)  # See emotions example
        photo_action.setStatusTip("See emotions example.")
        photo_action.triggered.connect(self.open_emotions_window)
        camera_toolbar.addAction(photo_action)

        # Create the worker Thread
        self.timer = QTimer()
        self.timer.setInterval(25)
        self.timer.timeout.connect(self.draw_camera)

        self.setWindowTitle("NSAViewer")
        self.show()

    def open_emotions_window(self):
        self.emotions_window = EmotionsWindow()
        self.emotions_window.show()
        return

    def show_graph(self):
        return

    def show_image(self):  # Load image from system
        filename = QFileDialog.getOpenFileName(self, 'Open file', resource_path('../Snapshots/'), "Image files (*.jpg *.gif *.png *.jpeg)")
        if filename[0] == "":  # Break from function if nothing is selected
            return

        self.image_frame = QLabel()
        cv_img = cv2.imread(filename[0])
        img_resized = cv2.resize(cv_img, (1024, 768))
        modif_img = detect_landmarks(img_resized)

        self.image = QImage(modif_img.data, modif_img.shape[1], modif_img.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(self.image))

        if self.currently_shown == "video":
            # self.camera.stop()
            self.stop_timer()
            self.currently_shown = "image"
            #self.viewfinder.hide()

        self.image_frame.show()
        self.setCentralWidget(self.image_frame)

    def take_photo(self):  # Snapshot
        if self.currently_shown != "video":
            return
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        ret, snapshot = self.camera.read()
        cv2.imwrite(resource_path(f"{self.save_path}{self.current_camera_name}-{self.save_seq}-{timestamp}.jpg"), snapshot)
        self.save_seq += 1

    def start_timer(self):  # Start live video timer
        if self.available_cameras:
            if self.currently_shown == "image":
                self.image_frame.hide()

            self.current_camera_name = self.available_cameras[0].description()
            self.currently_shown = "video"
            self.label = QLabel()
            self.setCentralWidget(self.label)
            self.timer.start()

    def stop_timer(self):  # Stop live video timer
        self.timer.stop()

    def draw_camera(self):  # Live video
        b, frame = self.camera.read()
        img_resized = cv2.resize(frame, (1024, 768))
        modif_img = detect_landmarks(img_resized)

        qImg = QImage(modif_img.data, modif_img.shape[1], modif_img.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(qImg)
        self.label.setPixmap(pix)

    def closeEvent(self, event):  # Event for closing video
        self.stop_timer()
        return QWidget.closeEvent(self, event)

    def change_folder(self):  # Change snapshot saving location
        path = QFileDialog.getExistingDirectory(self, "Snapshot save location", "")
        if path:
            self.save_path = path
            self.save_seq = 0

    def alert(self, s):  # Error handling
        err = QErrorMessage(self)
        err.showMessage(s)

    def start_video(self):  # Not used
        self.viewfinder = QCameraViewfinder()
        if self.currently_shown == "image":
            self.image_frame.hide()
            self.viewfinder.show()
        self.setCentralWidget(self.viewfinder)
        self.select_camera(0)

    def select_camera(self, i):  # Not used
        self.currently_shown = "video"
        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        self.camera.start()

        self.capture = QCameraImageCapture(self.camera)
        print(type(self.capture))
        self.capture.imageCaptured()
        self.capture.error.connect(lambda i, e, s: self.alert(s))
        self.capture.imageCaptured.connect(lambda d, i: self.status.showMessage("Image %04d captured" % self.save_seq))

        self.current_camera_name = self.available_cameras[i].description()
        self.save_seq = 0


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    init_facial_landmarks_detector()
    init_emotions_detector()

    app = QApplication(sys.argv)
    app.setApplicationName("NSAViewer")

    window = MainWindow()
    app.exec_()