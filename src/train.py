import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim

from torchvision.models import SqueezeNet
from model_backbones import CustomSqueezenet
from DataLoader import *
from tqdm import tqdm


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

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)


###### let's start training, bois #####

random_transforms = [transforms.ToPILImage(), 
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.RandomCrop(size=224)
]

print("Preparing data...")
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomApply(random_transforms, p=0.5)
    ]
)
dataset = EthnicityDataset('../data/dataset.csv', transform=transform)

train_set, validation_set = torch.utils.data.random_split(dataset, [965, 242])
train_data_loader = torch.utils.data.DataLoader(train_set, 
            batch_size=1, 
            shuffle=True)

test_data_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print("Done!")

EPOCHS = 4

print("Starting training...")
running_loss = 0.0
try:
    for e in tqdm(range(EPOCHS)):
        for i, batch in enumerate(train_data_loader):
            imgs = batch['image']
            labels = batch['label']
            print(f"Img type: {imgs.type} and shape: {imgs.shape}")
            print(f"Shape of labels: {labels.shape}")
            imgs = imgs.to(device)
            labels = (torch.Tensor(labels.float())).cuda()
            # print(labels)
            

            optimizer.zero_grad()

            outputs = model(imgs).cuda()

            # print(outputs.view(4,4))
            # exit()
            # print(f"Outputs: {outputs.view(1,4)}")
            # print(f'Labels: {labels}')
            # exit()
            loss = criterion(outputs.view(-1,4), labels.long())
            # loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print('[%d, %5d] loss: %f' %
                (e + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        torch.save(model, f'../models/squeezenet__1_1__4_classes_ epoch_{e}.pth')
        print(f"Saved model at epoch {e}")
# except Exception as e :
#     torch.save(model, 'squeezenet__1_1__4_classes_shit_happenend.pth')
#     print(e)
except KeyboardInterrupt:
    torch.save(model, '../models/squeezenet__1_1__4_classes_shit_happenend.pth')
    exit()

print("Done training!")
print("Starting testing....")

correct = 0

with torch.no_grad():
    for i, batch in tqdm(enumerate(test_data_loader)):
        img = batch['image']
        labels = batch['label']

        img = img.to(device)
        labels = (torch.Tensor(labels.float())).cuda()

        outputs = model(img)
        outputs = outputs.numpy()
        labels = labels.cpu()

        if np.argmax(labels.numpy()) == np.argmax(outputs):
            correct += 1


print(f"Test accuracy is: {correct / 242}")
    



