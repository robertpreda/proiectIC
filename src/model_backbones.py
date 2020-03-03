import torch
import torchvision
import torch.nn as nn

class CustomSqueezenet(nn.Module):
    def __init__(self, backbone, classes=4):
        super(CustomSqueezenet, self).__init__()
        self.classes = classes
        self.backbone = backbone
        final_conv = nn.Conv2d(512, self.classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            final_conv,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def get_model(model_path):

    if torch.cuda.is_available():
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    model.require_grad = False

    return model


# def get_predictions(model, input):
#     with torch.no_grad():


# class RacistNet(nn.Module):
#     def __init__(self, classes=4):
#         super(RacistNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3)
#         self.conv2 = nn.Conv2d()