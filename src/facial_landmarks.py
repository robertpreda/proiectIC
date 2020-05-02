import numpy as np
import imutils
import argparse
import dlib
import cv2
import torch
from src.utils import get_prediction

def rect_to_bb(rect):
	global width_coef, height_coef
	x = int(rect.left() * width_coef)
	y = int(rect.top() * height_coef)
	w = int(rect.right() * width_coef) - x
	h = int(rect.bottom() * height_coef) - y

	return x, y, w, h


def shape_to_np(shape, dtype="int"):
	global width_coef, height_coef
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (int(shape.part(i).x*width_coef), int(shape.part(i).y*height_coef))

	return coords


def detect_landmarks(image_inp):
	#image = cv2.imread(image_path)
	img_modif = image_inp
	image = imutils.resize(image_inp, width=512)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	# left_eye = [37, 38, 40, 41]
	# right_eye = [43, 44, 46, 47]
	for (i, rect) in enumerate(rects):

		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		# print(f"Difference between 40 and 37 in y is {shape[40][1] - shape[37][1]}")
		# print(f"Difference between 41 and 38 in y is {shape[41][1] - shape[38][1]}")
		# print(f"Difference between 46 and 43 in y is {shape[46][1] - shape[43][1]}")
		# print(f"Difference between 47 and 44 in y is {shape[47][1] - shape[44][1]}")
		med = (shape[40][1] - shape[37][1] + shape[41][1] - shape[38][1] + shape[46][1] - shape[43][1] + shape[47][1] - shape[44][1])/4
		(x, y, w, h) = rect_to_bb(rect)
		percentage = round((100*med)/h, 2)
		if percentage > 4:
			eyes = "Eyes opened"
		else:
			eyes = "Eyes closed"

		face = img_modif[y:y+h, x:x+w]
		face = cv2.resize(face, (256, 256))
		result = get_prediction(net, face, device).cpu()
		emotion = emotions[int(torch.argmax(result))]
		confidence = round(result.numpy()[0][int(torch.argmax(result))]*100,2)
		# print(emotion, confidence)

		cv2.rectangle(img_modif, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.putText(img_modif, f"#{i + 1} -- {eyes} -- {emotion} {confidence}%", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		for (x, y) in shape:
			cv2.circle(img_modif, (x, y), 1, (0, 0, 255), -2)
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)
	return img_modif


def init_facial_landmarks_detector():
	global detector, predictor, width_coef, height_coef, device
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	detector = dlib.get_frontal_face_detector()

	predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
	width_coef = 1024 / 512
	height_coef = 768 / 384


def init_emotions_detector():
	global net, emotions
	net = torch.load("../models/squeeze__epochs_100.pth")
	emotions = {
		0: 'Neutral',
		1: 'Happy',
		2: 'Sad',
		3: 'Surprise',
		4: 'Angry',
		5: 'Disgust',
		6: 'Fear'
	}


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
					help="path to input image")
	args = vars(ap.parse_args())

	width_coef = 1
	height_coef = 1

	init_facial_landmarks_detector()
	detect_landmarks(args["image"])

