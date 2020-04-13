import torch
import torch.nn as nn
import torchlayers

from torchtools.nn import Mish

class BaseCNN(nn.Module):
    '''
        TODO: make it work with any backbone, like a framework of some sorts
    '''
    def __init__(self, backone, classes):
        super(CNN, self).__init__()
        self.classes = classes
        self.backbone = backone
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, self.classes, kernel_size=1),
            Mish()
        )
    
    def forward(self, input_tensor):
        backbone_output = self.backone(input_tensor)
        classifier_output = self.classifier(backbone_output)

        return classifier_output

class nVGGCNN(nn.Module):
    """
        nVGG: novel-VGG (novel, because of the Mish activation function, original VGG uses ReLU)
    """
    def __init__(self, classes):
        super(nVGGCNN, self).__init__()
        self.classes = classes
        self.layers = [
            torchlayers.Conv(64),
            torchlayers.Mish(),
            torchlayers.Conv(128),
            torchlayers.Mish(),
            torchlayers.Conv(256),
            torchlayers.Mish(),
            torchlayers.BatchNorm(),
            torchlayers.pooling.AvgPool(),
            torchlayers.Linear(self.classes)
        ]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
