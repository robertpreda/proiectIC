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
from src.detect_faces import get_boxes


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model("../models/squeezenet__1_1__4_classes_ epoch_3__1582729000.pth")

race_info = [("#fffb6d", "Asian", "Corona"), ("#402D06", "Black", "Negro"), ("#c39752", "Latino", "Beaner"), ("#fef7d6", "White", "Gringo")]


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

        self.load_img_list_btn = tkinter.Button(self.window, text='Load Selected Image',
                                                command=self.load_selected_image)
        self.pack_place_obj(self.load_img_list_btn, height=25, width=150, x=15, y=375)

        self.load_directory_list_btn = tkinter.Button(self.window, text='Change dir', command=self.load_directory)
        self.pack_place_obj(self.load_directory_list_btn, height=25, width=150, x=15, y=15)

        self.canvas = tkinter.Canvas(self.window, width=1024, height=768)
        self.pack_place_obj(self.canvas, x=250, y=50)

        self.graph_canvas = tkinter.Canvas(self.window, width=400, height=700)

        self.load_img_btn = tkinter.Button(self.window, text='Load Image', command=self.open_img)
        self.pack_place_obj(self.load_img_btn, height=25, width=100, x=632, y=15)

        self.load_video_btn = tkinter.Button(self.window, text='Capture Video', command=self.capture_video)
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

    def draw_graph(self):
        self.graph_canvas.delete("all")
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
            self.graph_canvas.create_rectangle(x0, 400, x1, y, fill=race_info[i][0])

            # Put the y value above the bar
            self.graph_canvas.create_text(x0 + 2, y - 5, anchor=tkinter.SW, text=f"{round(self.data[0][i][0][0] * 100, 2)}%")
            self.graph_canvas.create_text(x0 + 2, 415, anchor=tkinter.SW, text=race_info[i][1])
            x_gap = x_gap + 20
        self.show_graph()

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
            resized_cv_img = cv2.resize(frame, (1024, 768))
            results, rgb_cv_image = get_resultimg_and_overallrace(resized_cv_img)

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rgb_cv_image))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            #results = run_model(frame)
            #self.data = F.softmax(results, dim=1).cpu().numpy()

            self.data = results
            self.draw_graph()
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
        ret, photo, results = self.img_obj.get_img(filename)
        self.data = results

        if ret:
            self.canvas.delete("all")
            self.hide_btn(self.btn_snapshot)
            self.hide_btn(self.what_am_i_btn)

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(photo))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.draw_graph()

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
            race = self.get_race()
            self.engine.say(f"You are a {race_info[race][2]}!")
            self.engine.runAndWait()
        except RuntimeError:
            return

    def get_race(self):
        race_indic = []
        for i in range(4):
            race_indic.append(self.data[0][i][0][0])
        race = np.argmax(race_indic)
        return race

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
    
    def show_graph(self):
        self.graph_canvas.pack()
        self.graph_canvas.place(bordermode=tkinter.OUTSIDE, x=1350, y=150)


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
                return ret, cv2.cvtColor(resized_cv_frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return False, None

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

        out_softed_tensor, rgb_cv_img = get_resultimg_and_overallrace(resized_cv_img)
        return True, rgb_cv_img, out_softed_tensor


def get_resultimg_and_overallrace(resized_cv_img):
    rects = get_boxes(resized_cv_img)
    out_softed_tensor = [[[[0]], [[0]], [[0]], [[0]]]]
    for x, y, w, h in rects:
        aux_img = resized_cv_img[y:y + h, x:x + w]

        # Uncomment if you want to see each face before processing
        #cv2.imshow("image", aux_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        tensor = run_model(aux_img)
        tensor_soft = F.softmax(tensor, dim=1).cpu().numpy()
        out_softed_tensor += tensor_soft
        index = np.argmax(tensor_soft)

        cv2.rectangle(resized_cv_img, (x, y), (x + w, y + h), color=(0, 255, 0),  thickness=1)  # Draw Rectangle with the coordinates
        cv2.putText(resized_cv_img, race_info[index][1] + " " + f"{round(tensor_soft[0][index][0][0] * 100, 2)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
    rgb_cv_img = cv2.cvtColor(resized_cv_img, cv2.COLOR_BGR2RGB)
    if len(rects) > 1:
        out_softed_tensor = out_softed_tensor / len(rects)
    return out_softed_tensor, rgb_cv_img


if __name__ == "__main__":
    # Create a window and pass it to the Application object
    MyApp = App(tkinter.Tk(), "Racisminator")
    cv2.destroyAllWindows()

