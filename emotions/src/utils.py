import torch
import torch.nn as nn
import torchtools
import torchvision.models as models
import torchvision.transforms as transforms

from torch.nn.functional import softmax

def get_resnet18(num_classes):
    new_layers = nn.Sequential(
        nn.Linear(1000, 256),
        nn.Linear(256, 128),
        nn.Linear(128, num_classes)
    )
    backbone = models.resnet50(pretrained=True)
    net = nn.Sequential(backbone, new_layers)
    return net

def get_squeezenet(num_classes):
    backbone = models.squeezenet1_1(pretrained=True)
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, 7, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(13)
    )
    return backbone

def get_prediction(network, input_data, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
        ]
    )
    face_tensor = transform(input_data)
    face_tensor = face_tensor.view(1, 3,256,256).float().to(device)
    with torch.no_grad():
        result = network(face_tensor).float()
        result.to('cpu')

    return softmax(result)


