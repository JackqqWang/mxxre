import torch
import torchvision.transforms.functional
from torch import nn



import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu


import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        
        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)        
        
        self.up_conv3 = double_conv(256 + 512, 256)
        self.up_conv2 = double_conv(128 + 256, 128)
        self.up_conv1 = double_conv(128 + 64, 64)
        
        self.final_conv = nn.Conv3d(64, 2, kernel_size=1)

    def forward(self, x, return_feature_vect = False):
        
        x1 = self.down_conv1(x)  
        x2 = self.maxpool(x1)    
        x2 = self.down_conv2(x2)
        x3 = self.maxpool(x2)    
        x3 = self.down_conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.down_conv4(x4)  
        
        
        x = F.interpolate(x4, size=x3.size()[2:], mode='trilinear', align_corners=True) 
        x = self.crop_and_concat(x3, x)
        x = self.up_conv3(x)
        
        x = F.interpolate(x, size=x2.size()[2:], mode='trilinear', align_corners=True) 
        x = self.crop_and_concat(x2, x)
        x = self.up_conv2(x)
        
        x = F.interpolate(x, size=x1.size()[2:], mode='trilinear', align_corners=True) 
        x = self.crop_and_concat(x1, x)
        x = self.up_conv1(x)
        
        
        if x.size()[2:] != x1.size()[2:]:
            x = F.interpolate(x, size=x1.size()[2:], mode='trilinear', align_corners=True)
        if return_feature_vect:
            return x
        x = self.final_conv(x)
        
        return x

    def crop_and_concat(self, bypass, upsampled):
        
        
        c_depth = (bypass.size()[2] - upsampled.size()[2]) // 2
        c_height = (bypass.size()[3] - upsampled.size()[3]) // 2
        c_width = (bypass.size()[4] - upsampled.size()[4]) // 2

        
        bypass_cropped = bypass[
            :,
            :,
            c_depth:bypass.size(2) - c_depth if (bypass.size()[2] - upsampled.size()[2]) % 2 == 0 else bypass.size(2) - c_depth - 1,
            c_height:bypass.size(3) - c_height if (bypass.size()[3] - upsampled.size()[3]) % 2 == 0 else bypass.size(3) - c_height - 1,
            c_width:bypass.size(4) - c_width if (bypass.size()[4] - upsampled.size()[4]) % 2 == 0 else bypass.size(4) - c_width - 1,
        ]

        
        return torch.cat([upsampled, bypass_cropped], dim=1)






import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class largeUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features = [64, 128, 256, 512, 1024]):
        super(largeUNet3D, self).__init__()
        
        self.encoders = nn.ModuleList(
            [double_conv(in_channels, features[0])] + 
            [double_conv(features[i], features[i+1]) for i in range(len(features)-1)]
        )

        self.decoders = nn.ModuleList(
            [double_conv(features[i], features[i-1]) for i in range(len(features)-1, 0, -1)]
        )

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsamples = nn.ModuleList(
            [nn.ConvTranspose3d(features[i], features[i-1], kernel_size=2, stride=2) for i in range(len(features)-1, 0, -1)]
        )
        
        self.bottleneck = double_conv(features[-2], features[-1])
        
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x, return_feature_vect = False):
        skip_connections = []

        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.encoders[-1](x)
        
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.decoders)):
            x = self.upsamples[idx](x)
            skip_connection = skip_connections[idx]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=True)
            
            x = torch.cat((x, skip_connection), dim=1)
            x = self.decoders[idx](x)
        if return_feature_vect:
            return x

        return self.final_conv(x)























class PrivateClassifier_seg(nn.Module):
    def __init__(self):
        super(PrivateClassifier_seg, self).__init__()
        self.outconv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        return self.linear(x)
    
class PublicClassifier_seg(nn.Module):
    def __init__(self):
        super(PublicClassifier_seg, self).__init__()
        self.outconv = nn.Conv2d(128, 1, kernel_size=1)








class largeUnet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        
        self.e11 = nn.Conv2d(3, 128, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
        self.e13 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e21 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 
        self.e23 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e31 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
        self.e32 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.e33 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e41 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) 
        self.e42 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) 
        self.e43 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) 
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e51 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1) 
        self.e52 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
        self.e53 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1) 

        
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.d13 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) 

        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d23 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.d33 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d43 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 

        self.outconv = nn.Conv2d(128, n_class, kernel_size=1)

    def forward(self, x, return_feature_vect=False):
        
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xe13 = relu(self.e13(xe12)) 
        xp1 = self.pool1(xe13)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xe23 = relu(self.e23(xe22)) 
        xp2 = self.pool2(xe23)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xe33 = relu(self.e33(xe32)) 
        xp3 = self.pool3(xe33)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xe43 = relu(self.e43(xe42)) 
        xp4 = self.pool4(xe43)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        xe53 = relu(self.e53(xe52)) 

        
        xu1 = self.upconv1(xe53)
        xu11 = torch.cat([xu1, xe43], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))
        xd13 = relu(self.d13(xd12)) 

        xu2 = self.upconv2(xd13)
        xu22 = torch.cat([xu2, xe33], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))
        xd23 = relu(self.d23(xd22)) 

        xu3 = self.upconv3(xd23)
        xu33 = torch.cat([xu3, xe23], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))
        xd33 = relu(self.d33(xd32)) 

        xu4 = self.upconv4(xd33)
        xu44 = torch.cat([xu4, xe13], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))
        xd43 = relu(self.d43(xd42)) 
        if return_feature_vect:
            return xd43

        out = self.outconv(xd43)

        return out










class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        
        
        
        
        
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) 
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) 


        
        
        
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x, return_feature_vect = False):
        
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))
        if return_feature_vect:
            return xd42

        
        out = self.outconv(xd42)
        

        return out