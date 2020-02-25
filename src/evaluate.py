import cv2
import torch
import numpy as np
import pyttsx3

from torchvision import transforms
from model_backbones import CustomSqueezenet

engine = pyttsx3.init()
engine.setProperty('rate', 100)
engine.setProperty('volume', 0.7)

MODEL_PATH = '../models/squeezenet__1_1__4_classes_ epoch_3.pth'
IMAGE_PATH = 'me.jpg'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


labels = {
    0: 'Coronovirus',
    1: 'Negro',
    2: 'Beanle',
    3: 'Gringo'
}

model = torch.load(MODEL_PATH)
model.require_grad = False

img_orig = cv2.imread(IMAGE_PATH)
img = cv2.resize(img_orig, (550, 550))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_img = transform(img).view(-1, 3, img.shape[0], img.shape[1]).float().to(device)
with torch.no_grad():
    results = model(input_img)
    print(results)
    res = np.argmax(results.cpu().detach().numpy())



window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
org = (50, 50)
thickness = 2
color = (255, 0, 0) 
detection = labels[res]
image = cv2.putText(img_orig, detection, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow('Image', image)


############# VOICES ##############

# TODO: get some racist lines for each races
# TODO: randomly select a line for each time a race is being detected

# if detection == 'Coronavirus':
#     engine.say('Ching chong, do your homework')
#     engine.runAndWait()
# elif detection == 'Negro':
#     engine.say("Please don't steal my bike, Ape")
#     engine.runAndWait()
# elif detection == 'Beanle':
#     engine.say("Shit, I need to build a higher fence")
#     engine.runAndWait()
# elif detection == 'Gringo':
#     engine.say("Gringo, got some meth?")
#     engine.runAndWait()
# if cv2.waitKey(0) & 0xff == 27:
#     exit() 
# print(f"Dis a {labels[res]}")