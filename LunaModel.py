import torch.nn as nn
import numpy as np 


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super(LunaBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        
        return self.maxpool(block_out)



class LunaModel(nn.Module):
    def __init__(self,in_channels = 1, conv_channels= 8): 
        super(LunaModel, self).__init__()

        self.tail_layer = nn.BatchNorm3d(in_channels)

        self.block_1 = LunaBlock(in_channels, conv_channels)
        self.block_2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block_3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block_4 = LunaBlock(conv_channels * 4, conv_channels * 8)  
        # note how progressively the number of channels increase 

        # Remember that the input shape is (32, 48, 48) (depth, height, width)
        # due to maxpooling layer at the end of each block
        # 48 -> 24 -> 12 -> 6 -> 3 (spatial)
        # 32 -> 16 -> 8 -> 4 -> 2  (depth)
        self.head_linear = nn.Linear(2 * 3 * 3 * 8 * conv_channels, 2) 
        self.softmax_head = nn.Softmax(dim = 1) 
        self.dropout = nn.Dropout(0.3)


    def _init_weights(self):
        for m in self.modules():
            if type(m) in  {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode = "fan_out", nonlinearity="relu") # He initialization 
                if m.bias is not None: # can be neglected as the biases don't impose any troubles in the data flow through the network
                    _ , fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1/ np.sqrt(fan_out) # xavior initialization for biases 
                    nn.init.normal_(m.bias,-bound, bound)


    def forward(self, input_batch):
        bn_output = self.tail_layer(input_batch)

        block_out = self.block_1(bn_output)
        block_out = self.block_2(block_out)
        block_out = self.block_3(block_out)
        block_out = self.block_4(block_out)
        conv_flatten = block_out.view(
            block_out.size(0), # batch_size
            -1
        )
        dropout_out = self.dropout(conv_flatten) 
        linear_output = self.head_linear(dropout_out)
        return linear_output, self.softmax_head(linear_output)
            


