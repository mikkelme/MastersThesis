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
from torchinfo import summary

import numpy as np
  

class VGGNet(Module):
    """ VGGNet style CNN 

    Init Args:
        mode (int): Toggle different type of data flow. 
            0: Image and numerical values goes through convolution no seperate input channels
            1: Image get convoluted while numerical values is inserted at FC
        out_features (list of strings): Define the output features and their type. 'R' refer to regression type quanity (continous) and 'C' refers to a classification type feature (sigmoid to [0, 1])
        input_num (int): Number of numerical values as input (e.g. stretch and F_N) in addition to config matrix
        conv_layers (list of tuples): Define architecture of convolution part. Each convolution block is defined by the tuple and is repeated tuple[0] times seperated by a RELU activation and gets tuple[1] channels each time. Each block is then seperated by a maxpooling layer.
        FC_layers (list of tuples): Define architecture of fully connected (FC) part. Each fully connected block is repeated tuple[0] times with a total of tuple[1] nodes. Each FC layer is seperated by a RELU activation except the last one which is handled by the hardcoded output layers

    """        
    def __init__( self, name = 'VGGNet',
                        mode = 0, 
                        image_shape = (62, 106), 
                        input_num = 2,
                        conv_layers = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
                        FC_layers   = [(2, 4096)],
                        out_features = ['R', 'C'],
                        keys = None,
                        batchnorm = True):
       
       
        super(VGGNet, self).__init__()
        
        
        
        if keys is not None:
            assert len(keys) == len(out_features), f"Number of keys {len(keys)}: {keys}, does not match number of out_features {len(out_features)}: {out_features}"
       
        self.name = name
        self.input_num = input_num
        self.image_shape = image_shape
        self.conv_layers = conv_layers
        self.FC_layers = FC_layers
        self.out_features = np.array(out_features)
        self.keys = keys
        self.mode = mode
        self.batchnorm = batchnorm
        
        if mode == 0: # channels for each numerical input
            numChannels = 1 + input_num
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
                
                if batchnorm:
                    self.layers.append(BatchNorm2d(num_features = filter[1]))
                self.layers.append(ReLU())
                prev_channels = filter[1]
            self.layers.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding = 1))
            prev_shape = (prev_shape[0]//2 + 1, prev_shape[1]//2 + 1)
        self.len_conv = len(self.layers)
        
        # --- Fully connected (FC) block --- #
        prev_features = prev_channels * prev_shape[0] * prev_shape[1] 
        if mode == 1:
            prev_features += input_num
            
        for i, filter in enumerate(FC_layers):
            for j in range(filter[0]):
                self.layers.append(Linear(in_features=prev_features, out_features=filter[1]))
                self.layers.append(ReLU())
                prev_features = filter[1]
        self.layers.pop(-1)
        
        
        # --- FC Output --- #
        self.fc = Linear(in_features=prev_features, out_features=len(out_features))
        self.sigmoid = Sigmoid()
        self.sig_map = self.out_features == 'C'
        


        # --- Initialize weights --- #
        # self.apply(self. init_weights)
        # exit()
        
        # placeholder for the gradients
        self.gradients = None
        # self.hook_layer = self.len_conv - 2
        # self.hook_layer = self.len_conv - 2 -4*4
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad


    # method for the activation exctraction
    def get_activations(self, x, hook_layer): # Assumes f_mix XXX
        image, vals = x
        
        # --- Gather input into channels --- #
        # Unsqueeze numerical valuess and expand channels into mathing image dimensions
        vals = vals.view(vals.shape[0], vals.shape[1], 1, 1).expand(vals.shape[0], vals.shape[1], image.shape[1], image.shape[2])
        
        # Unsqueeze image for concatenation
        image = torch.unsqueeze(image, dim = 1)
        
        # Concatenate input channels 
        x = torch.cat((image, vals), dim = 1)
        
        for l in range(hook_layer+1):
            x = self.layers[l](x)
        return x


    def f_mix(self, x, hook_layer = None):
        """ Image and numerical input (on indivual channels) all go through convolution """
        image, vals = x
        
        # --- Gather input into channels --- #
        # Unsqueeze numerical valuess and expand channels into mathing image dimensions
        vals = vals.view(vals.shape[0], vals.shape[1], 1, 1).expand(vals.shape[0], vals.shape[1], image.shape[1], image.shape[2])
        
        # Unsqueeze image for concatenation
        image = torch.unsqueeze(image, dim = 1)
        
        # Concatenate input channels 
        x = torch.cat((image, vals), dim = 1)
        
        # --- Forward pass --- #
        # Convolutional 
        for l in range(self.len_conv):
            x = self.layers[l](x)
            
            if hook_layer:
                if l == hook_layer:  # register hook for grad-cam
                    h = x.register_hook(self.activations_hook)
        
        # Fully connected (FC)
        x = flatten(x, 1)
        for l in range(self.len_conv, len(self.layers)):
            x = self.layers[l](x)
        
        # Output
        x = self.fc(x)
        x[:,self.sig_map] = self.sigmoid(x[:,self.sig_map]) # sigmoid for classfication type variables
         
        return x
        
    def f_insert(self, x):
        """ Image through convolution and numerical values input directly to FC """
        
        image, vals = x
        
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
        x[:,self.sig_map] = self.sigmoid(x[:,self.sig_map]) # sigmoid for classfication type variables

        
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
            
    
    def get_num_params(self):
        num_param = 0
        for param in self.parameters():
            p = torch.prod(torch.tensor(param.size()))
            num_param += p
        return num_param.item()
        
        
    def __str__(self):
        s = summary(self, [(1, self.image_shape[0], self.image_shape[1]), (1, self.input_num)],
                    verbose = 0)
        return str(s)
        
        
    
        
        
        
if __name__ == '__main__':
    # model = VGGNet(mode = 0, image_shape = (62, 106), input_num = 2)
    
    model = VGGNet( mode = 0, 
                    input_num = 2, 
                    image_shape = (62, 106),
                    conv_layers = [(1, 16), (1, 32)], 
                    FC_layers = [(1, 32), (1, 16)],
                    out_features = ['R', 'C'],
                    batchnorm = True)
    
    num_params = model.get_num_params()
    print(num_params)
    # print(model)
    
   