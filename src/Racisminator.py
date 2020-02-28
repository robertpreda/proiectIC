import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import pyttsx3
import os
import torch
import torch.nn.functional as F
import numpy as np
from src.model_backbones import get_model
from tkinter import filedialog
from threading import Thread
from torchvision import transforms

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model("../models/squeezenet__1_1__4_classes_ epoch_3__1582729000.pth")


def run_model(cv_img):
    cv_model_resized = cv2.resize(cv_img, (550, 550))
    with torch.no_grad():
        tensor = model(transform(cv_model_resized).view(-1, 3, 550, 550).float().to(device))
    return tensor


class App:
    def __init__(self, window, window_title, video_source=0):
        self.engine = pyttsx3.init()
        # Set properties _before_ you add things to say
        self.engine.setProperty('rate', 150)  # Speed percent (can go over 100)
        self.engine.setProperty('volume', 1)  # Volume 0-1

        self.window = window
        self.window.iconphoto(True, PIL.ImageTk.PhotoImage(file="../Resources/Doofenshmirtz_Portrait.png"))
        self.window.geometry("1700x900+50+50")
        self.window.resizable(width=True, height=True)
        self.window.title(window_title)
        self.video_source = video_source
        self.delay = 15
        self.running = ""

        self.listbox = tkinter.Listbox(self.window)
        self.listbox.pack()
        self.listbox.place(bordermode=tkinter.OUTSIDE, height=300, width=200, x=15, y=100)
        self.filelist = []
        self.get_filelist()

        self.load_img_list_btn = tkinter.Button(self.window, text='Load Selected Image',
                                                command=self.load_selected_image)
        self.load_img_list_btn.pack()
        self.load_img_list_btn.place(bordermode=tkinter.OUTSIDE, height=25, width=150, x=15, y=425)

        self.canvas = tkinter.Canvas(self.window, width=1024, height=768)
        self.canvas.pack()
        self.canvas.place(bordermode=tkinter.OUTSIDE, x=250, y=15)

        self.graph_canvas = tkinter.Canvas(self.window, width=400, height=700)
        self.race_info = [("#fffb6d", "Asian", "Corona"), ("#402D06", "Black", "Negro"), ("#c39752", "Latino", "Beaner"), ("#fef7d6", "White", "Gringo")]

        self.load_img_btn = tkinter.Button(self.window, text='Load Image', command=self.open_img)
        self.load_img_btn.pack()
        self.load_img_btn.place(bordermode=tkinter.OUTSIDE, height=25, width=100, x=15, y=15)

        self.load_video_btn = tkinter.Button(self.window, text='Capture Video', command=self.capture_video)
        self.load_video_btn.pack()
        self.load_video_btn.place(bordermode=tkinter.OUTSIDE, height=25, width=100, x=15, y=45)

        self.what_am_i_btn = tkinter.Button(self.window, text='What am I?', command=self.what_am_i)
        self.hide_btn(self.what_am_i_btn)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.img_obj = MyLoadImg()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.hide_btn(self.btn_snapshot)

        self.window.mainloop()

    def draw_graph(self):
        self.graph_canvas.delete("all")
        #self.data = np.random.uniform(low=0, high=1, size=(4,))

        # The variables below size the bar graph
        x_width = 40  # The width of the x-axis
        x_gap = 0  # The gap between left canvas edge and y axi
        for i in range(4):
            # Bottom left coordinate
            x0 = (i + 1) * x_width + x_gap
            # Bottom right coordinates
            x1 = (i + 2) * x_width + x_gap
            # Top coordinates
            y = 400 - 400 * self.data[0][i][0][0]
            # Draw the bar
            self.graph_canvas.create_rectangle(x0, 400, x1, y, fill=self.race_info[i][0])

            # Put the y value above the bar
            self.graph_canvas.create_text(x0 + 2, y - 5, anchor=tkinter.SW, text=f"{round(self.data[0][i][0][0] * 100, 2)}%")
            self.graph_canvas.create_text(x0 + 2, 415, anchor=tkinter.SW, text=self.race_info[i][1])
            x_gap = x_gap + 20
        self.show_graph()

    def what_am_i(self):
        self.talk_thread = Thread(target=self.talk)
        try:
            self.talk_thread.start()
        except RuntimeError:
            self.talk_thread.join()

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
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            results = run_model(frame)

            self.data = F.softmax(results, dim=1).cpu().numpy()
            self.draw_graph()

        if self.running == "Video":
            self.window.after(self.delay, self.update)

    def load_selected_image(self):
        x = self.listbox.curselection()[0]
        self.open_img(f"../Snapshots/{self.listbox.get(x)}")

    def open_img(self, filename=""):
        self.talk_thread = Thread(target=self.talk)
        self.running = "Img"
        ret, photo, results = self.img_obj.get_img(filename)

        self.data = F.softmax(results, dim=1).cpu().numpy()

        if ret:
            self.hide_btn(self.btn_snapshot)
            self.hide_btn(self.what_am_i_btn)
            self.canvas.delete("all")

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(photo))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.draw_graph()

            self.talk_thread.start()
        return

    def capture_video(self):
        self.running = "Video"
        self.show_btn(self.btn_snapshot, 25, 100, 712, 800)
        self.show_btn(self.what_am_i_btn, 25, 100, 600, 800)
        self.canvas.delete("all")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.update()
        return

    def get_filelist(self):
        self.listbox.delete(0, tkinter.END)
        self.filelist = os.listdir("../Snapshots")
        for file in self.filelist:
            self.listbox.insert(tkinter.END, file)

    def hide_btn(self, btn):
        btn.pack_forget()
        btn.place_forget()

    def show_btn(self, btn, height, width, x, y):
        btn.pack(anchor=tkinter.CENTER, expand=True)
        btn.place(bordermode=tkinter.OUTSIDE, height=height, width=width, x=x, y=y)

    def show_graph(self):
        self.graph_canvas.pack()
        self.graph_canvas.place(bordermode=tkinter.OUTSIDE, x=1350, y=150)

    def talk(self):
        race_indic = []
        try:
            for i in range(4):
                race_indic.append(self.data[0][i][0][0])
            race = np.argmax(race_indic)
            self.engine.say(f"You are a {self.race_info[race][2]}!")
            self.engine.runAndWait()
        except RuntimeError:
            return


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            resized_cv_frame = cv2.resize(frame, (1024, 768))
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(resized_cv_frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)

    # Release the video source when the object is destroyed
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
        resized_cv_img = cv2.resize(cv_img, (1024, 768))

        tensor = run_model(cv_img)

        rgb_cv_img = cv2.cvtColor(resized_cv_img, cv2.COLOR_BGR2RGB)

        return True, rgb_cv_img, tensor


# Create a window and pass it to the Application object
MyApp = App(tkinter.Tk(), "Racisminator")
cv2.destroyAllWindows()

