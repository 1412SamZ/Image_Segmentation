import torch
from torch import nn
import torchvision.transforms.functional as TF 

class DoubleConv(nn.Module):
    
    #Repeated double CNN
    def __init__(self, in_channels, out_channels, batchNorm=nn.BatchNorm2d ,activation_fn=nn.ReLU):
        super().__init__()
        self.batchNorm = batchNorm
        self.activation_fn = activation_fn
        
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            self.batchNorm(out_channels),
            self.activation_fn(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            self.batchNorm(out_channels),
            self.activation_fn(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
    
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        # Downwards part
        for feat in features:
            self.down.append(DoubleConv(in_channels, feat))
            in_channels = feat 
            
        # Upwards part
        for feat in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2)
            )
            self.up.append(DoubleConv(feat*2, feat))
            
        # between down and up 
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # the final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        for dn in self.down:
            x = dn(x)
            skip_connections.append(x)
        
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.up), 2):
            
            # upsample
            x = self.up[idx](x)
            
            # get the result from down-step
            skip_connection = skip_connections[idx//2]
            
            # to match the size of the to be concatinated parts
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            
            # concatinate these 2 parts
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # let the concatinated input go through double cnn
            x = self.up[idx+1](concat_skip)
            
        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 1, 203, 107))
    model = Unet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    
test()