import torch
import torch.nn as nn
from .ssa import SoftmaxSamplingActivation

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, nclass, activation='relu', tau=1.0, hard=False):
        super(VGG, self).__init__()
        self.activation_type = activation
        self.tau = tau
        self.hard = hard
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.activation_type == 'relu':
                    activation = nn.ReLU(inplace=True)
                elif self.activation_type == 'ssa':
                    activation = SoftmaxSamplingActivation(tau=self.tau, hard=self.hard)
                else:
                    raise ValueError(f"Unknown activation: {self.activation_type}")
                
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           activation]
                if self.activation_type == 'ssa':
                    layers += [nn.BatchNorm2d(x)]
                in_channels = x
        return nn.Sequential(*layers)
