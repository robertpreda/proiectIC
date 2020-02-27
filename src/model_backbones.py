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

    model = torch.load(model_path)
    model.require_grad = False

    return model

# class RacistNet(nn.Module):
#     def __init__(self, classes=4):
#         super(RacistNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3)
#         self.conv2 = nn.Conv2d()