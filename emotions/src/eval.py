import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import numpy as np
import torchvision.models as models

from architectures import nVGGCNN
from backbones import SqueezeNet

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        
    ]
)

emotions = {
    0 : 'Neutral',
    1 : 'Happy',
    2 : 'Sad',
    3 : 'Surprise',
    4 : 'Angry',
    5 : 'Disgust',
    6 : 'Fear'
}

parser = argparse.ArgumentParser()
parser.add_argument('--input',dest='input_data')
parser.add_argument('--model', dest='model')

face_detector = cv2.CascadeClassifier('../models/face_classifier.xml')

args = parser.parse_args()
# net = models.squeezenet1_1(num_classes=7)
state_dict = torch.load(args.model)
net = torch.load(args.model)
input_source = args.input_data if args.input_data else 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net.to(device)

if input_source == 0:
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        faces = face_detector.detectMultiScale(img, scaleFactor=1.5,minNeighbors=4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        for x,y,w,h in faces:
            
            try:
                temp_face = img[x:x+w, y:y+h]
                resized = cv2.resize(temp_face, (256,256))
            except:
                continue
            # resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # img2 = np.zeros_like(resized)
            # img2[:,:,0] = resized_gray
            # img2[:,:,1] = resized_gray
            # img2[:,:,2] = resized_gray
            # # resized_gray = img2
            face_tensor = transform(resized)
            # face_tensor = torch.from_numpy(resized)
            face_tensor = face_tensor.view(1, 3,256,256).float().to(device)
            
            with torch.no_grad():
                result = net(face_tensor).float()
                result.to('cpu')
            print(result)
            prediction = emotions[int(torch.argmax(result))]
            cv2.rectangle(img, (x,y),(x+w, y+h),(0,0,255),2)
            cv2.putText(img, prediction,(x,y),font,fontScale,color,thickness,cv2.LINE_AA)

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xff == ord('q'):
                exit()

else:
    img = cv2.imread(args.input_data)
    faces = face_detector.detectMultiScale(img, scaleFactor=1.1,minNeighbors=4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    for x,y,w,h in faces:
        temp_face = img[x:x+w, y:y+h]
        try:
            resized = cv2.resize(temp_face, (256,256))
        except:
            print(temp_face.shape)
            continue
        # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        face_tensor = transform(resized)
        face_tensor = face_tensor.view(1, 3,256,256).float().to(device)
        result = net(face_tensor).float()
        result.to('cpu')
        # print(torch.argmax(result))
        print(F.softmax(result))
        prediction = emotions[int(torch.argmax(F.softmax(result)))]
        cv2.rectangle(img, (x,y),(x+w, y+h),(0,0,255),2)
        cv2.putText(img, prediction,(x,y),font,fontScale,color,thickness,cv2.LINE_AA)

        cv2.imshow("Image", img)

        if cv2.waitKey(0) & 0xff == ord('q'):
            exit()


