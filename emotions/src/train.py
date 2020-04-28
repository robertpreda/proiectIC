import torch
import argparse
import torchvision
import torchtools
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



from tqdm import tqdm
from torchtools.optim import RangerLars
from architectures import nVGGCNN
from CustomDataLoader import DataLoaderFacesDB as face_dataloader
from CustomDataLoader import DataSetFromCSV
from backbones import SqueezeNet

from utils import get_resnet18
from utils import get_squeezenet

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

models_ = {
    'vgg':nVGGCNN,
    'squeeze': SqueezeNet
}

def accuracy(output, labels):
    return torch.mean(torch.argmax(output, dim=1).eq(torch.argmax(labels, dim=1)).type(torch.FloatTensor))

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', dest='model_name')
parser.add_argument('--batch_size', dest='batch_size',default=1)
parser.add_argument('--epochs', dest='epochs',default=1)
parser.add_argument('--lr', dest='lr', default=1e-5)
parser.add_argument('--save_folder', dest='folder')

args = parser.parse_args()

net = get_squeezenet(7)
# net.classifier = nn.Sequential(
#     nn.Dropout(p=0.5),
#     nn.Conv2d(512, 7, kernel_size=1),
#     nn.ReLU(inplace=True),
#     nn.AvgPool2d(13)
# )
# net.forward = lambda x: net.classifier(net.features(x)).view(x.size(0), 7)


net.to(device)

# net = torchlayers.build(net, torch.randn(1,3,48,48))

criterion = nn.CrossEntropyLoss().cuda()
# optimizer = torchtools.optim.RangerLars(net.parameters(), lr=float(args.lr))
optimizer = torchtools.optim.RAdam(net.parameters(), lr=float(args.lr), weight_decay=0.001)

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        
    ]
)



dataset = DataSetFromCSV('facesdb.csv', transform=transform)
train_size = int(len(dataset) * 0.8 )
train_set, validation_set = torch.utils.data.random_split(dataset, [train_size, int(len(dataset) -train_size)])
batch_size = int(args.batch_size)

print(f"Batch size of {batch_size}")

data_loader = torch.utils.data.DataLoader(train_set, 
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
)
test_data_loader = torch.utils.data.DataLoader(validation_set, 
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
)

EPOCHS = int(args.epochs)
try:
    print("Starting training...")
    running_loss = 0.0
    list_epoch_accuracies = []
    for e in tqdm(range(EPOCHS)):
        for i, batch in enumerate(data_loader):
            imgs = batch['image'].view(batch_size,3,256,256).float()
            labels = torch.Tensor(batch['label'].float())
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            # zero out the gradients
            optimizer.zero_grad()

            prediction = net(imgs).to(device)
            prediction = prediction.view(batch_size, 7)
            # print(f"Prediction = {prediction}, Label = {labels}")
            # exit()
            batch_accuracy = accuracy(prediction, labels)
            list_epoch_accuracies.append(batch_accuracy)
            loss = criterion(prediction, torch.argmax(labels.float(), dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 19:

                print(f"Epoch {e} Iteration {i} Loss: {loss.item()} \t Batch accuracy {batch_accuracy}")
            running_loss = 0.0
        
       
        epoch_accuracy = np.mean(list_epoch_accuracies)
        print(f"Epoch accuracy: {epoch_accuracy}")
except KeyboardInterrupt:
    torch.save(net, f'../models/{args.model_name}__interrupt__epoch_{e}.pth')
    exit()

torch.save(net, f'../models/{args.folder}/{args.model_name}__epochs_{EPOCHS}.pth')
print(f"Saved model at epoch {e}")
############## testing #############
net.eval()
acc_list = []
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_data_loader)):
        img = batch['image']
        labels = batch['label']

        img = img.to(device)
        labels = (torch.Tensor(labels.float())).cuda()

        outputs = net(img)
        outputs = F.softmax(outputs.cpu())
        labels = labels.cpu()

        acc = accuracy(outputs, labels)
        acc_list.append(acc)
        print(f"Batch {i} accuracy: {acc} %")

print(f"Test accuracy: {np.mean(acc_list)}")