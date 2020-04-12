import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pyttsx3

from torchvision import transforms
from model_backbones import CustomSqueezenet
from model_backbones import get_model

engine = pyttsx3.init()
engine.setProperty('rate', 100)
engine.setProperty('volume', 0.7)

# F:\Cursuri\AN 3\sem2\IC\Proiect\models\squeezenet__1_1__4_classes_epoch_3__1583795360_new_dataset.pth

MODEL_PATH = '../models/squeezenet__1_1__4_classes_epoch_3__1583795360_new_dataset.pth'
IMAGE_PATH = 'me.jpg'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


labels = {
    0: 'Asian',
    1: 'Black',
    2: 'Latino',
    3: 'White'
}

model = get_model(MODEL_PATH)



img_orig = cv2.imread(IMAGE_PATH)
img = cv2.resize(img_orig, (550, 550))

transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_img = transform(img).view(1, 3, 550, 550).float().to(device)
with torch.no_grad():
    
    results = model(input_img)

    res_sorted = np.sort(np.array(results.cpu().numpy()))

    res = np.argmax(results.cpu().detach().numpy())

# text_with_percentages = f"{labels[0]}: {res_sorted[0][0]} \n {labels[1]} : {res_sorted[0][1]} \n {labels[2]} : {res_sorted[0][2]} \n {labels[3]} : {res_sorted[0][3]} \n"



window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 0.75
org = (50, int(img_orig.shape[0]/2))
thickness = 1
color = (255, 0, 0) 
detection = labels[res] + ": " + str(results[0][res][0][0].cpu().numpy() * 100) + "%"



print(F.softmax(results).cpu().numpy()[0])
image = cv2.putText(img_orig, labels[res], org, font,  
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
if cv2.waitKey(0) & 0xff == 27:
    exit() 
# print(f"Dis a {labels[res]}")