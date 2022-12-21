from torch import nn
from torchsummary import summary
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')


class CNN_ResNet():
    def __init__(self):
        self.resnet_model = resnet34(pretrained=True)
        self.resnet_model.fc = nn.Linear(512, 4)
        self.resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # resnet_model = resnet_model.to(device)

    def GetModel(self):
        return self.resnet_model



if __name__ == "__main__":
    resnet = CNN_ResNet().GetModel().to(device)
    summary(resnet.cuda(), (1, 64, 431))

