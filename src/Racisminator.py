import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import pyttsx3
import os
import torch
from src.facial_landmarks import init_facial_landmarks_detector, detect_landmarks
from tkinter import filedialog, PhotoImage, Label
from threading import Thread
from torchvision import transforms


class App:
    def __init__(self, window, window_title, video_source=0):
        self.engine = pyttsx3.init()
        # Set properties _before_ you add things to say
        self.engine.setProperty('rate', 150)  # Speed percent (can go over 100)
        self.engine.setProperty('volume', 1)  # Volume 0-1

        self.window = window
        self._callback_id = tkinter.StringVar(self.window)
        self._callback_id.set(None)
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.window.iconphoto(True, PIL.ImageTk.PhotoImage(file="../Resources/Doofenshmirtz_Portrait.png"))
        self.window.geometry("1700x900+50+50")
        self.window.resizable(width=True, height=True)
        self.window.title(window_title)
        self.video_source = video_source
        self.delay = 10
        self.data = []
        self.dirname = "../Snapshots"

        self.listbox = tkinter.Listbox(self.window)
        self.pack_place_obj(self.listbox, height=300, width=200, x=15, y=50)
        self.filelist = []
        self.get_filelist()

        self.load_img_list_btn = tkinter.Button(self.window, text='Load Selected Image', command=self.load_selected_image)
        self.pack_place_obj(self.load_img_list_btn, height=25, width=150, x=15, y=375)

        self.load_directory_list_btn = tkinter.Button(self.window, text='Change dir', command=self.load_directory)
        self.pack_place_obj(self.load_directory_list_btn, height=25, width=150, x=15, y=15)

        self.canvas = tkinter.Canvas(self.window, width=1024, height=768)
        self.pack_place_obj(self.canvas, x=250, y=50)

        self.load_img_btn = tkinter.Button(self.window, text='Load Image', command=self.open_img)
        self.pack_place_obj(self.load_img_btn, height=25, width=100, x=632, y=15)

        self.load_video_btn = tkinter.Button(self.window, text='Video', command=self.capture_video)
        self.pack_place_obj(self.load_video_btn, height=25, width=100, x=800, y=15)

        self.what_am_i_btn = tkinter.Button(self.window, text='What am I?', command=self.what_am_i)
        self.hide_btn(self.what_am_i_btn)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.img_obj = MyLoadImg()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.hide_btn(self.btn_snapshot)

        self.window.mainloop()

    def what_am_i(self):
        talk_thread = Thread(target=self.talk)
        try:
            talk_thread.start()
        except RuntimeError:
            talk_thread.join()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("../Snapshots/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.get_filelist()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            rgb_cv_image = detect_landmarks(frame)

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rgb_cv_image))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self._callback_id.set(self.window.after(self.delay, self.update))

    def load_selected_image(self):
        try:
            self.window.after_cancel(self._callback_id.get())
            x = self.listbox.curselection()[0]
            self.open_img(f"{self.dirname}/{self.listbox.get(x)}")
        except IndexError:
            return

    def open_img(self, filename=""):
        self.window.after_cancel(self._callback_id.get())
        talk_thread = Thread(target=self.talk)
        ret, photo = self.img_obj.get_img(filename)

        if ret:
            self.canvas.delete("all")
            self.hide_btn(self.btn_snapshot)
            self.hide_btn(self.what_am_i_btn)

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(photo))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            talk_thread.start()
        return

    def capture_video(self):

        self.show_btn(self.btn_snapshot, height=25, width=100, x=712, y=830)
        self.show_btn(self.what_am_i_btn, height=25, width=100, x=600, y=830)
        self.canvas.delete("all")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.update()
        return

    def talk(self):
        try:
            self.engine.say("Ahahahaha")
            self.engine.runAndWait()
        except RuntimeError:
            return

    def get_filelist(self):
        self.listbox.delete(0, tkinter.END)
        self.filelist = os.listdir(self.dirname)
        for file in self.filelist:
            self.listbox.insert(tkinter.END, file)

    def load_directory(self):
        self.dirname = filedialog.askdirectory()
        self.get_filelist()

    def hide_btn(self, btn):
        btn.pack_forget()
        btn.place_forget()

    def show_btn(self, btn, height, width, x, y):
        btn.pack(anchor=tkinter.CENTER, expand=True)
        btn.place(bordermode=tkinter.OUTSIDE, height=height, width=width, x=x, y=y)
    
    def pack_place_obj(self, obj,  x, y, height=-1, width=-1):
        if height == -1 and width == -1:
            obj.place(bordermode=tkinter.OUTSIDE, x=x, y=y)
        else:
            obj.place(bordermode=tkinter.OUTSIDE, height=height, width=width, x=x, y=y)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            resized_cv_frame = cv2.resize(frame, (1024, 768))
            if ret:
                return ret, cv2.cvtColor(resized_cv_frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return False, None

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class MyLoadImg:
    def __init__(self):
        self.filename = ""

    def get_img(self, filename):
        if filename == "":
            self.filename = filedialog.askopenfilename(title='open')
            if self.filename == "":
                return False, None, None
        else:
            self.filename = filename

        cv_img = cv2.imread(self.filename)
        rgb_cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(rgb_cv_img, (1024, 768))
        modif_img = detect_landmarks(img_resized)
        return True, modif_img


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    init_facial_landmarks_detector()

    # Create a window and pass it to the Application object
    MyApp = App(tkinter.Tk(), "Racisminator")
    cv2.destroyAllWindows()

