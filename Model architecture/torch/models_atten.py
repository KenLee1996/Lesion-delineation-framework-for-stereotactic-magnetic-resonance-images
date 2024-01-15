import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_block(nn.Module):
    def __init__(self, g_size, x_size, out_size):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(g_size, out_size, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm3d(out_size)
            #nn.GroupNorm(8, out_size))
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(x_size, out_size, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm3d(out_size)
            #nn.GroupNorm(8, out_size))
        )

        self.psi = nn.Sequential(
            nn.Conv3d(out_size, 1, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class DP_atten_resUNet(nn.Module):
    def __init__(self, img_channels = 2, n_classes = 2):
        super(DP_atten_resUNet, self).__init__()
        
        self.pool = nn.MaxPool3d((2, 2, 1), stride = (2, 2, 1))
        
        self.conv_xy_0 = nn.Sequential(
            nn.Conv3d(img_channels, 16, kernel_size = (3, 3, 1), padding = (1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16))
        self.conv_z_0 = nn.Sequential(
            nn.Conv3d(img_channels, 16, kernel_size = (1, 1, 3), padding = (0, 0, 1)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16))

        self.conv_xy_1 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size = (3, 3, 1), padding = (1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16),
            nn.Conv3d(16, 16, kernel_size = (3, 3, 1), padding = (1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16))
        self.conv_z_1 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size = (1, 1, 3), padding = (0, 0, 1)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16),
            nn.Conv3d(16, 16, kernel_size = (1, 1, 3), padding = (0, 0, 1)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16))
        
        self.conv_xy_2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size = (3, 3, 1), padding = (1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 32))
        self.conv_z_2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size = (1, 1, 3), padding = (0, 0, 1)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 32))

        self.conv_xy_3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size = (3, 3, 1), padding = (1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 32),
            nn.Conv3d(32, 32, kernel_size = (3, 3, 1), padding = (1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 32))
        self.conv_z_3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size = (1, 1, 3), padding = (0, 0, 1)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 32),
            nn.Conv3d(32, 32, kernel_size = (1, 1, 3), padding = (0, 0, 1)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 32))
        
        self.conv_decode_1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 64),
            nn.Conv3d(64, 32, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 32),
            nn.ConvTranspose3d(32, 16, kernel_size = 3, stride = (2, 2, 1), padding = 1, output_padding=(1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16))
        
        self.conv_decode_2 = nn.Sequential(
            nn.Conv3d(48, 24, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 24),
            nn.ConvTranspose3d(24, 16, kernel_size = 3, stride = (2, 2, 1), padding = 1, output_padding=(1, 1, 0)),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16))
        
        self.conv_decode_3 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.GroupNorm(8, 16),
            nn.Conv3d(16, n_classes, kernel_size = 1, padding = 0),
            )
        
        self.atten_1x1_32 = Attention_block(16, 32, 32)
        self.atten_1x1_16 = Attention_block(16, 16, 16)
        self.conv_1x1_32 = nn.Conv3d(64, 32, kernel_size = 1, padding = 0)
        self.conv_1x1_16 = nn.Conv3d(32, 16, kernel_size = 1, padding = 0)
        
    def forward(self, img_input):

        xy_0 = self.conv_xy_0(img_input)
        z_0 = self.conv_z_0(img_input)        
        xy_1 = self.conv_xy_1(xy_0)
        z_1 = self.conv_z_1(z_0)
        xy_1_out = xy_0+xy_1
        z_1_out = z_0+z_1

        xy_2 = self.conv_xy_2(self.pool(xy_1_out))
        z_2 = self.conv_z_2(self.pool(z_1_out))        
        xy_3 = self.conv_xy_3(xy_2)
        z_3 = self.conv_z_3(z_2)
        xy_3_out = xy_2+xy_3
        z_3_out = z_2+z_3
        
        decode_1 = self.conv_decode_1(torch.cat([self.pool(xy_3_out), self.pool(z_3_out)], dim = 1))

        xyz_2 = self.conv_1x1_32(torch.cat([xy_3_out, z_3_out], dim = 1))
        xyz_2 = self.atten_1x1_32(decode_1, xyz_2)
        #xyz_2 = self.atten_1x1_32(decode_1, torch.cat([xy_3_out, z_3_out], dim = 1))

        decode_2 = self.conv_decode_2(torch.cat([xyz_2, decode_1], dim = 1))
        
        xyz_1 = self.conv_1x1_16(torch.cat([xy_1_out, z_1_out], dim = 1))
        xyz_1 = self.atten_1x1_16(decode_2, xyz_1)
        #xyz_1 = self.atten_1x1_16(decode_2, torch.cat([xy_1_out, z_1_out], dim = 1))
        
        decode_3 = self.conv_decode_3(torch.cat([xyz_1, decode_2], dim = 1))
            
        output = torch.sigmoid(decode_3)
        #output = torch.softmax(decode_3,dim=1)
        #output = decode_3

        return output