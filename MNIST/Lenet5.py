import torch
import torch.nn as nn
from utils import Binarize, Conv2dBinary, LinearBinary

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False),#mudar o canal de entrada pare 3 caso usar a cifar10
            nn.BatchNorm2d(6),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )

        self.linearLayer = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.linearLayer(out)
        return out

class LeNet5XNOR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
 
            # nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=True),
            Binarize(),
            Conv2dBinary(in_channels=20, out_channels=50, kernel_size=5),
            nn.Hardtanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.dense = nn.Sequential(
            # nn.BatchNorm1d(50*5*5),
            Binarize(),
            LinearBinary(50*5*5, 500),
            nn.Hardtanh(),
        )

        self.fc = nn.Linear(500, num_classes)


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)

            # if isinstance(m, Conv2dBinary) or isinstance(m, LinearBinary):
            #     m.weight.register_post_accumulate_grad_hook(updateBinaryGradWeight)

        

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)

        out = self.features(x)
        # out = self.layer3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        out = self.fc(out)
        return out
