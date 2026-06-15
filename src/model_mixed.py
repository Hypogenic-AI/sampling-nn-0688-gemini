import torch
import torch.nn as nn
from .ssa import SoftmaxSamplingActivation

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class VGG_Mixed(nn.Module):
    def __init__(self, vgg_name, nclass, tau=1.0, hard=False):
        super(VGG_Mixed, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], tau, hard)
        self.classifier = nn.Linear(512, nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, tau, hard):
        layers = []
        in_channels = 3
        # Only use SSA for the LAST feature layer
        num_layers = len([x for x in cfg if x != 'M'])
        layer_idx = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layer_idx += 1
                if layer_idx == num_layers:
                    activation = SoftmaxSamplingActivation(tau=tau, hard=hard)
                    print(f"Using SSA for layer {layer_idx}")
                else:
                    activation = nn.ReLU(inplace=True)
                
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           activation]
                in_channels = x
        return nn.Sequential(*layers)
