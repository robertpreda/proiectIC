import torch
import torch.nn as nn
import torch.nn.functional as F
import torchlayers

# from torchtools.activations import Mish

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        if torch.cuda.is_available():
            return x.view(batch_size, -1).cuda()
        else:
            return x.view(batch_size, -1)

class BaseCNN(nn.Module):
    '''
        TODO: make it work with any backbone, like a framework of some sorts
    '''
    def __init__(self, backone, classes):
        super(BaseCNN, self).__init__()
        self.classes = classes
        self.backbone = backone
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, self.classes, kernel_size=1),
            torchlayers.nn.Mish()
        )
    
    def forward(self, input_tensor):
        backbone_output = self.backone(input_tensor)
        classifier_output = self.classifier(backbone_output)

        return classifier_output

class nVGGCNN(nn.Module):

    def __init__(self, classes):
        super(nVGGCNN, self).__init__()
        self.classes = classes
        self.nn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64,64,kernel_size=3),
            nn.MaxPool2d((2,2)),
            # torchlayers.BatchNorm(),
            # torchlayers.pooling.AvgPool(),
            nn.Flatten(),
            nn.Linear(25600, 512),
            nn.Linear(512, 256),
            nn.Linear(256,128),
            nn.Linear(128, self.classes),
            nn.ReLU()
        )
    def forward(self, x):
        for layer in self.nn_layers:
            x = layer(x)
        return x
