import torch
import torch.nn as nn
from Encoder_Decoder import Encoder, Decoder
from feature_transforms import wct

def stylize(level, content, style, encoders, decoders, alpha, svd_device, cnn_device):

    with torch.no_grad():
        cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
        sf = encoders[level](style).data.to(device=svd_device).squeeze(0)
        
        csf = wct(alpha, cf, sf).to(device=cnn_device)

    return decoders[level](csf)

class WCTmodel(nn.Module):

    def __init__(self, args):
        super(WCTmodel, self).__init__()

        self.svd_device = torch.device('cpu')
        self.cnn_device = args.device
        self.alpha = args.alpha

        self.e1 = Encoder(1)
        self.e2 = Encoder(2)
        self.e3 = Encoder(3)
        self.e4 = Encoder(4)
        self.e5 = Encoder(5)
        self.encoders = [self.e5, self.e4, self.e3, self.e2, self.e1]

        self.d1 = Decoder(1)
        self.d2 = Decoder(2)
        self.d3 = Decoder(3)
        self.d4 = Decoder(4)
        self.d5 = Decoder(5)
        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]

    def forward(self, content, style):

        for i in range(len(self.encoders)):
            content = stylize(i, content, style, self.encoders, self.decoders, 
                            self.alpha, self.svd_device, self.cnn_device)

        return content