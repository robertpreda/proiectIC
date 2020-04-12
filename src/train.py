import torch
import time
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from torchvision.models import SqueezeNet
from model_backbones import CustomSqueezenet
from DataSet import *
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def accuracy(output, labels):
    # print(f"Outputs: {output}")
    # print(f"Labels: {labels}")

    return torch.mean(torch.argmax(output, dim=1).eq(torch.argmax(labels, dim=1)).type(torch.FloatTensor)) 
    


# sn_model = SqueezeNet(version='1_1', num_classes=4)

print('Loading SqueezeNet....')
sn_model = torch.load('../models/squeezenet__1_1__4_classes.pth')

model = CustomSqueezenet(sn_model)

print('Done!')

# print('Freezing layers...')
# i = 0
# for module in model.modules():
#     i += 1
#     if i < 6:
#         for param in module.parameters():
#             param.requires_grad = False
# print('Done!')
class_weights = [514/109, 1, 514/106, 514/447]
class_weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
random_transforms = [transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomRotation(45)
]

###### let's start training, bois #####

print("Preparing data...")
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply(random_transforms, p=0.75),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
    ]
)
dataset = EthnicityDataset('./dataset_v2.csv', transform=transform)

batch_size = 8

# train_set, validation_set = torch.utils.data.random_split(dataset, [941, 236])
train_data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=True, drop_last=True)
# test_data_loader = torch.utils.data.DataLoader(validation_set, 
#             batch_size=batch_size, 
#             shuffle=True, drop_last=True)



model.to(device)
print("Done!")

EPOCHS = 4

print("Starting training...")
running_loss = 0.0
# correct = 0
try:
    for e in tqdm(range(EPOCHS)):
        
        for i, batch in enumerate(train_data_loader):
            imgs = batch['image']
            labels = batch['label']

            imgs = imgs.to(device)
            labels = (torch.Tensor(labels.float())).cuda()
            # print(labels)
            

            optimizer.zero_grad()

            outputs = model(imgs).cuda()

            # print(outputs)
            # exit()
            # print(f"Outputs: {outputs.size()}")
            # print(f'Labels: {labels.size()}')
            # exit()
            outputs = outputs.view(batch_size, 4)
            batch_accuracy = accuracy(outputs, labels)
            loss = criterion(outputs, torch.argmax(labels.float(), dim=1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # print('[%d, %5d] loss: %f' %
            #     (e + 1, i + 1, running_loss / 2000))
            print(f"Epoch {e} Iteration {i} Loss: {running_loss} \t Batch accuracy: {batch_accuracy}")
            running_loss = 0.0

        torch.save(model, f'../models/squeezenet__1_1__4_classes_epoch_{e}__{int(time.time())}_new_dataset.pth')
        print(f"Saved model at epoch {e}")
# except Exception as e :
#     torch.save(model, 'squeezenet__1_1__4_classes_shit_happenend.pth')
#     print(e)
except KeyboardInterrupt:
    torch.save(model, '../models/squeezenet__1_1__4_classes_shit_happenend.pth')
    exit()

print("Done training!")
print("Starting testing....")

# acc_list = []
# with torch.no_grad():
#     for i, batch in tqdm(enumerate(test_data_loader)):
#         img = batch['image']
#         labels = batch['label']

#         img = img.to(device)
#         labels = (torch.Tensor(labels.float())).cuda()

#         outputs = model(img)
#         outputs = outputs.cpu()
#         labels = labels.cpu()

#         acc = accuracy(outputs, labels)
#         acc_list.append(acc)
#         print(f"Batch {i} accuracy: {acc} %")

# print(f"Test accuracy: {np.mean(acc_list)}")



