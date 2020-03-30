import numpy as np
import imutils
import argparse
import dlib
import cv2

def rect_to_bb(rect):
	global width_coef, height_coef
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = int(rect.left() * width_coef)
	y = int(rect.top() * height_coef)
	w = int(rect.right() * width_coef) - x
	h = int(rect.bottom() * height_coef) - y
	# return a tuple of (x, y, w, h)
	return x, y, w, h


def shape_to_np(shape, dtype="int"):
	global width_coef, height_coef
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (int(shape.part(i).x*width_coef), int(shape.part(i).y*height_coef))
	# return the list of (x, y)-coordinates
	return coords


def detect_landmarks(image_inp):
	#image = cv2.imread(image_path)
	img_modif = image_inp
	image = imutils.resize(image_inp, width=512)
	print(image_inp.shape)
	print(image.shape)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = rect_to_bb(rect)
		cv2.rectangle(img_modif, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the face number
		cv2.putText(img_modif, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(img_modif, (x, y), 1, (0, 0, 255), -2)
	# show the output image with the face detections + facial landmarks
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)
	return img_modif


def init_facial_landmarks_detector():
	global detector, predictor, width_coef, height_coef
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	# predictor = dlib.shape_predictor(args["shape_predictor"])
	predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
	width_coef = 1024 / 512
	height_coef = 768 / 384


if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	#  ap.add_argument("-p", "--shape-predictor", required=True,
	#  				help="path to facial landmark predictor")
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
					help="path to input image")
	args = vars(ap.parse_args())

	width_coef = 1
	height_coef = 1

	init_facial_landmarks_detector()
	detect_landmarks(args["image"])

