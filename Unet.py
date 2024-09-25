import torch 
import torch.nn as nn 
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, in_channel,
                num_classes = 2,
                resolution_levels = 5,
                filters_power =  6,
                padding = False, 
                batch_norm = False, 
                up_mode = "learnable"):
        
        """
        Implementation of the Unet architecture for biomedical segmentation.

        Parameters:
            - in_channels: the number of input channels.
            - num_classes: number of output classes of semantic segmentation. 
            - resolution_levels: the depth of the U network.
            - filters_power: power factor for computing the starting
            - channel expansion (number of filters of first conv is 2**(filters_power)).
            - padding: specify whether you want the input spatial dim is as the output or not. 
            - batch_norm: specify the possibility of applying batch normalization after network layers.
            - up_mode: "learnable" -> deconvoluation  or "non-learnable" -> bilinear upsampling. 
        """
        super(UNet, self).__init__()
        assert up_mode.lower() in ("learnable", "non-learnable")
        self.padding = padding
        self.depth = resolution_levels
        
        # model initialization  
        self.contraction_path = nn.ModuleList()
        self.expansion_path = nn.ModuleList()

        # initialize the contraction path 
        prev_channels = in_channel 
        for i in range(resolution_levels): # going downstair
            self.contraction_path.append(UNetConvBlock(prev_channels, 2**(i+filters_power), padding, batch_norm))
            prev_channels = 2**(i+filters_power) # for subsequent conv block input channels 

        for i in reversed(range(resolution_levels-1)): # going upstair  
            self.expansion_path.append(UNetUpBlock(prev_channels, 2**(i+filters_power), up_mode, padding, batch_norm))
            prev_channels = 2**(i+filters_power)


        self.last_layer = nn.Conv2d(prev_channels, num_classes, kernel_size=1) # point-wise conv for projection


    def forward(self, x):
        for_skips = list()
        for i, down_block in enumerate(self.contraction_path):
            x = down_block(x)
            if i != (len(self.depth) - 1):
                for_skips.append(x) # for the upward path concatenation
                F.max_pool2d(x, 2)
        
        for i, up_block in enumerate(self.expansion_path):
            x = up_block(x, for_skips[-1 - i]) 

        return self.last_layer(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        seq_layers = list()

        seq_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)))
        seq_layers.append(nn.ReLU())
        if batch_norm:
            seq_layers.append(nn.BatchNorm2d(out_channels))

        seq_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)))
        seq_layers.append(nn.ReLU())
        if batch_norm:
            seq_layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*seq_layers)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode.lower() == "learnable":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode.lower() == "non-learnable":
            self.up = nn.Sequential(nn.Upsample(mode="bilinear", scale_factor=2), 
                                    nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # IMPORTANT: the input channels hasn't changed even we have performed a up step due to the skip concat   
        self.conv_block = UNetConvBlock(in_channels, out_channels, padding, batch_norm) 

    def center_crop(self, layer, target_size):
        """
        Centering the crop and the upsampled output for concatenation. This is especially use in case of non-padding.
        """
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
    

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out