import torch
import argparse
import torchlayers
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchtools.optim import RangerLars
from architectures import nVGGCNN
from CustomDataLoader import DataLoaderFacesICML2013 as face_dataloader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

models = {
    'vgg':nVGGCNN
}

def accuracy(output, labels):
    return torch.mean(torch.argmax(output, dim=1).eq(torch.argmax(labels, dim=1)).type(torch.FloatTensor))

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', dest='model_name')

args = parser.parse_args()




net = models[args.model_name](classes=7)
# net = torchlayers.build(net, torch.randn(1,3,48,48))

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters())

dataset = face_dataloader('../data/fer2013.csv')
batch_size = 8

data_loader = torch.utils.data.DataLoader(dataset, 
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
)
net.cuda()
EPOCHS = 4
try:
    print("Starting training...")
    running_loss = 0.0
    for e in tqdm(range(EPOCHS)):
        for i, batch in enumerate(data_loader):
            imgs = batch['image'].view(-1,1,48,48).float()
            labels = (torch.Tensor(batch['label'].float()))

            imgs = imgs.to(device)
            labels = labels.to(device)
            # zero out the gradients
            optimizer.zero_grad()

            prediction = net(imgs)
            prediction = prediction.view(batch_size, 7).cuda()
            # print(f"Prediction = {prediction}, Label = {labels}")
            # exit()
            batch_accuracy = accuracy(prediction, labels)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Epoch {e} Iteration {i} Loss: {running_loss} \t Batch accuracy {batch_accuracy}")
            running_loss = 0.0
        torch.save(net, f'../models/{args.model_name}__epoch_{e}.pth')
        print(f"Saved model at epoch {e}")
except KeyboardInterrupt:
    torch.save(net, f'../models/{args.model_name}__interrupt__epoch_{e}.pth')
    exit()
