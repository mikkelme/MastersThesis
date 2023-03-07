# import torch
# import torch.nn as nn
# from RainforestDataset import get_classes_list

import torch
import torch.nn as nn
from torch.nn import Module # Base class for all neural network modules.
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import LogSoftmax
from torch.nn import BatchNorm2d

from torch import flatten
from torchsummary import summary

class LeNet(Module):
    def __init__(self, numChannels):
        super(LeNet, self).__init__()
        
        # Inialize 1. set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(11, 11))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        
        # Inialize 2. set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(4, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
		# Initialize FC ==> RELU layers
        self.fc1 = Linear(in_features=2500, out_features=500)
        # self.fc1 = Linear(in_features=2500, out_features=500)
        self.relu3 = ReLU()
        
        # Intialize FC output
        self.fc2 = Linear(in_features=500, out_features=2)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        # x-shape = (1-3, 63 x 106)
    
        # 1. set
        x = self.conv1(x) # (20, 52 x 96)
        x = self.relu1(x)
        x = self.maxpool1(x) # (20, 13 x 24)
        
        # 2. set
        x = self.conv2(x) # (50, 10 x 20)
        x = self.relu2(x)
        x = self.maxpool2(x) # (50, 5 x 10 )
        
        # 3. set
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        # Output
        x = self.fc2(x)
        x[:,1] = self.sigmoid(x[:,1]) # sigmoid for is_ruptured
        
        return x
        
  

class VGGNet(Module):
    """ VGGNet style CNN 

    Init Args:
        mode (int): Toggle different type of data flow. 
            0: Image and numerical values goes through convolution no seperate input channels
            1: Image get convoluted while numerical values is inserted at FC
        num_vals (int): Number of numerical values as input (e.g. stretch and F_N)
        conv_layers (list of tuples): Define architecture of convolution part. Each convolution block is defined by the tuple and is repeated tuple[0] times seperated by a RELU activation and gets tuple[1] channels each time. Each block is then seperated by a maxpooling layer.
        FC_layers (list of tuples): Define architecture of fully connected (FC) part. Each fully connected block is repeated tuple[0] times with a total of tuple[1] nodes. Each FC layer is seperated by a RELU activation except the last one which is handled by the hardcoded output layers

    """        
    def __init__(self,  mode = 0, 
                        image_shape = (62, 106), 
                        num_vals = 2,
                        out_features = 2,
                        conv_layers = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
                        FC_layers   = [(2, 4096)]):
       
        super(VGGNet, self).__init__()
        
        self.image_shape = image_shape
        
        
        if mode == 0: # channels for each numerical input
            numChannels = 1 + num_vals
            self.forward = self.f_mix
        if mode == 1: # numerical inputs to FC directly
            numChannels = 1
            self.forward = self.f_insert
        
        self.layers = nn.ModuleList()
        
        # --- Convolutional blocks --- #
        prev_channels = numChannels
        prev_shape = self.image_shape
        for i, filter in enumerate(conv_layers):
            for j in range(filter[0]):
                self.layers.append(Conv2d(in_channels=prev_channels, 
                                          out_channels=filter[1], 
                                          kernel_size=(3, 3), 
                                          padding = 'same', 
                                          stride = 1))
                
                self.layers.append(BatchNorm2d(num_features = filter[1]))
                self.layers.append(ReLU())
                prev_channels = filter[1]
            self.layers.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding = 1))
            prev_shape = (prev_shape[0]//2 + 1, prev_shape[1]//2 + 1)
        self.len_conv = len(self.layers)
        
        # --- Fully connected (FC) block --- #
        prev_features = prev_channels * prev_shape[0] * prev_shape[1] 
        if mode == 1:
            prev_features += num_vals
            
        for i, filter in enumerate(FC_layers):
            for j in range(filter[0]):
                self.layers.append(Linear(in_features=prev_features, out_features=filter[1]))
                self.layers.append(ReLU())
                prev_features = filter[1]
        self.layers.pop(-1)
        
        
        # --- FC Output --- #
        self.fc = Linear(in_features=prev_features, out_features=out_features)
        self.sigmoid = Sigmoid()


        # --- Initialize weights --- #
        # self.apply(self. init_weights)
        # exit()

    def f_mix(self, image, vals):
        """ Image and numerical input (on indivual channels) all go through convolution """
        # Gather input into channels
        x = [image]
        for i in range(vals.size(-1)):
            x.append(torch.stack([torch.full(self.image_shape, v) for v in vals[:, i]], 0))
        x = torch.stack(x, 1)

        
        # Convolutional 
        for l in range(self.len_conv):
            x = self.layers[l](x)
        
        # Fully connected (FC)
        x = flatten(x, 1)
        for l in range(self.len_conv, len(self.layers)):
            x = self.layers[l](x)
        
        # Output
        x = self.fc(x)
        x[:,-1] = self.sigmoid(x[:,-1]) # sigmoid for is_ruptured
        # print(self.fc.weight)
        
        return x
        
    def f_insert(self, image, vals):
        """ Image through convolution and numerical values input directly to FC """
        x = torch.unsqueeze(image, 1)
        
        # Convolutional 
        for l in range(self.len_conv):
            x = self.layers[l](x)
        
        # Fully connected (FC)
        x = flatten(x, 1)
        x = torch.cat((x, vals), 1)
        for l in range(self.len_conv, len(self.layers)):
            x = self.layers[l](x)
        
        # Output
        x = self.fc(x)
        x[:,-1] = self.sigmoid(x[:,-1]) # sigmoid for is_ruptured
        
        return x
    
    
    def init_weights(self, module):
        
         if isinstance(module, Conv2d) or isinstance(module, Linear) :
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        
        
        # if isinstance(module, Conv2d):
        #     # print(module)
        #     # print(torch.mean(module.weight).item(), torch.std(module.weight).item())
        #     torch.nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
        #     # print(torch.mean(module.weight).item(), torch.std(module.weight).item())
        #     # print()
        #     if module.bias is not None:
        #         nn.init.constant_(module.bias, 0.01)
                
        # elif isinstance(module, Linear):
        #     # if module.out_features == 3:
        #     #     pass
        #     # else:
        #     module.weight.data.normal_(mean=0.0, std=0.01)
        #     nn.init.constant_(module.bias, 0.01)
        #         # torch.nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
            
            
            
            # module.weight.data.normal_(mean=0.0, std=0.1)
            # if module.bias is not None:
            #     nn.init.constant_(module.bias, 0.01)
                
        
        # if isinstance(module, nn.Linear):
        #     module.weight.data.normal_(mean=0.0, std=1.0)
        #     if module.bias is not None:
        #         nn.init.constant_(module.bias, 0.01)
            
    
        # elif isinstance(module, BatchNorm2d):
        #     nn.init.constant_(module.weight, 1) 
        #     nn.init.constant_(module.bias, 0) 
            
    
    
    
    # def __str__(self):
    #     s = '#---- LAYERS ---- #\n'
    #     for l in self.layers:
    #         s += str(l)
    #         s += '\n'
    #     s += '-- OUTPUT -- \n'
    #     s += str(self.fc)
    #     s += '\n'
    #     s += str(self.sigmoid)
    #     s += '\n'
    #     return s             
        
        
        
if __name__ == '__main__':
    model = VGGNet(mode = 0, image_shape = (62, 106), num_vals = 2)
    pass
    # model = LeNet(3)
    # print(model)
    # for param in model.parameters():
    #     print(type(param.data), param.size())
    
    
    
    # print(summary(LeNet(3), (3, 62, 106)))
    # print(summary(VGGNet(), ([62, 106], [2])))
    
    # print(summary(VGGNet(), [(62, 106), (2)]))
    
    
