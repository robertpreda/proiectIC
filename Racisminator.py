import tkinter
import numpy as np
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk
import time
import pyttsx3
from threading import Thread, enumerate

class App:
    def __init__(self, window, window_title, video_source=0):
        self.engine = pyttsx3.init()
        # Set properties _before_ you add things to say
        self.engine.setProperty('rate', 150)  # Speed percent (can go over 100)
        self.engine.setProperty('volume', 1)  # Volume 0-1

        self.window = window
        self.window.iconphoto(True, PIL.ImageTk.PhotoImage(file="Doofenshmirtz_Portrait.png"))
        self.window.geometry("1700x900+50+50")
        self.window.resizable(width=True, height=True)
        self.window.title(window_title)
        self.video_source = video_source
        self.delay = 15
        self.running = ""

        self.canvas = tkinter.Canvas(self.window, width=1024, height=768)
        self.canvas.pack()
        self.canvas.place(bordermode=tkinter.OUTSIDE, x=200, y=15)

        self.graph_canvas = tkinter.Canvas(self.window, width=400, height=500)
        self.race_info = [("#fffb6d", "Asian"), ("#402D06", "Black"), ("#c39752", "Latino"), ("#fef7d6", "White")]

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
        self.data = np.random.uniform(low=0, high=1, size=(4,))

        # The variables below size the bar graph
        x_width = 40  # The width of the x-axis
        x_gap = 0  # The gap between left canvas edge and y axi
        for i in range(len(self.data)):
            # Bottom left coordinate
            x0 = (i+1) * x_width + x_gap
            # Bottom right coordinates
            x1 = (i+2) * x_width + x_gap
            # Top coordinates
            y = 400 - 400 * self.data[i] + 15

            # Draw the bar
            self.graph_canvas.create_rectangle(x0, 400, x1, y, fill=self.race_info[i][0])

            # Put the y value above the bar
            self.graph_canvas.create_text(x0 + 2, y-5, anchor=tkinter.SW, text=f"{round(self.data[i] * 100, 2)}%")
            self.graph_canvas.create_text(x0 + 2, 415, anchor=tkinter.SW, text=self.race_info[i][1])
            x_gap = x_gap + 20
        self.show_graph()

    def what_am_i(self):
        self.talk_thread = Thread(target=self.talk)
        print("You a nigga!")
        #self.draw_graph()
        try:
            self.talk_thread.start()
        except RuntimeError:
            self.talk_thread.join()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        self.draw_graph()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        if self.running == "Video":
            self.window.after(self.delay, self.update)

    def open_img(self):
        self.talk_thread = Thread(target=self.talk)
        self.running = "Img"
        ret, photo = self.img_obj.get_img()

        if ret:
            self.hide_btn(self.btn_snapshot)
            self.hide_btn(self.what_am_i_btn)
            self.canvas.delete("all")

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(photo))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.draw_graph()

            self.talk_thread.start()
        print(enumerate())
        return

    def capture_video(self):
        self.running = "Video"
        self.show_btn(self.btn_snapshot, 25, 100, 712, 800)
        self.show_btn(self.what_am_i_btn, 25, 100, 600, 800)
        self.canvas.delete("all")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.update()
        return

    def hide_btn(self, btn):
        btn.pack_forget()
        btn.place_forget()

    def show_btn(self, btn, height, width, x, y):
        btn.pack(anchor=tkinter.CENTER, expand=True)
        btn.place(bordermode=tkinter.OUTSIDE, height=height, width=width, x=x, y=y)

    def show_graph(self):
        self.graph_canvas.pack()
        self.graph_canvas.place(bordermode=tkinter.OUTSIDE, x=1300, y=100)

    def talk(self):
        try:
            self.engine.say("You are a race!")
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

    def get_img(self):
        self.filename = filedialog.askopenfilename(title='open')
        if self.filename == "":
            return False, None

        cv_img = cv2.imread(self.filename)
        resized_cv_img = cv2.resize(cv_img, (1024, 768))
        rgb_cv_img = cv2.cvtColor(resized_cv_img, cv2.COLOR_BGR2RGB)

        return True, rgb_cv_img


# Create a window and pass it to the Application object
MyApp = App(tkinter.Tk(), "Racisminator")
cv2.destroyAllWindows()

