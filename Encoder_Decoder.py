import torch.nn as nn
import torch

class Encoder(nn.Module):

    def __init__(self, depth):
        super(Encoder, self).__init__()

        self.depth = depth

        if depth == 1:#conv2
            self.model = nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),
            )
            self.model.load_state_dict(torch.load("models/vgg_normalised_conv1_1.pth"))
        elif depth == 2:#conv4
            self.model = nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),
            )
            self.model.load_state_dict(torch.load("models/vgg_normalised_conv2_1.pth"))
        elif depth == 3:#conv6
            self.model = nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU(),
            )
            self.model.load_state_dict(torch.load("models/vgg_normalised_conv3_1.pth"))
        elif depth == 4:#conv10
            self.model = nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 512, (3, 3)),
                nn.ReLU(),
            )
            self.model.load_state_dict(torch.load("models/vgg_normalised_conv4_1.pth"))
        elif depth == 5:#conv14
            self.model = nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 512, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(), 
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(), 
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(), 
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),
            )
            self.model.load_state_dict(torch.load("models/vgg_normalised_conv5_1.pth"))
        
    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(nn.Module):

    def __init__(self, depth):
        super(Decoder, self).__init__()

        self.depth = depth

        if depth == 1:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 3, (3, 3)),
            )
            self.model.load_state_dict(torch.load("models/feature_invertor_conv1_1.pth"))
        elif depth == 2:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 64, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 3, (3, 3)),
            )
            self.model.load_state_dict(torch.load("models/feature_invertor_conv2_1.pth"))
        elif depth == 3:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 128, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 64, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 3, (3, 3)),
            )
            self.model.load_state_dict(torch.load("models/feature_invertor_conv3_1.pth"))
        elif depth == 4:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 256, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 128, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 64, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 3, (3, 3)),
            )
            self.model.load_state_dict(torch.load("models/feature_invertor_conv4_1.pth"))
        elif depth == 5:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 256, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 128, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 64, (3, 3)),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 3, (3, 3)),
            )
            self.model.load_state_dict(torch.load("models/feature_invertor_conv5_1.pth"))
    
    def forward(self, x):
        return self.model(x)