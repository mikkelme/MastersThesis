# import torch
# import torch.nn as nn
# from RainforestDataset import get_classes_list


from torch.nn import Module # Base class for all neural network modules.
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import LogSoftmax
from torch import flatten


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
        
  


# class Identity(nn.Module):
#     """
#     Identity class to return the input of the previous

#     functions: forward
#     """
#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, x):
#         return x


# class TwoNetworks(nn.Module):
#     """
#     This class takes two pretrained networks,
#     concatenates the high-level features before feeding these into
#     a linear layer.

#     functions: forward
#     """

#     def __init__(self, pretrained_net1, pretrained_net2):
#         super(TwoNetworks, self).__init__()

#         _, num_classes = get_classes_list()
#         num_ftrs1 = pretrained_net1.fc.in_features
#         num_ftrs2 = pretrained_net2.fc.in_features

#         pretrained_net1.fc = Identity()
#         pretrained_net2.fc = Identity()
#         self.fully_conv1 = pretrained_net1 # rgb
#         self.fully_conv2 = pretrained_net2 # infrared

#         new_in_channels = 1
#         layer = self.fully_conv2.conv1

#         # Creating new Conv2d layer
#         new_layer = nn.Conv2d(in_channels=new_in_channels,
#                           out_channels=layer.out_channels,
#                           kernel_size=layer.kernel_size,
#                           stride=layer.stride,
#                           padding=layer.padding,
#                           bias=layer.bias)

#         # Copy weights from red color channel (probably best choice for infrared channel)
#         with torch.no_grad():
#             new_layer.weight[:,:,:,:] = layer.weight[:, :1, :, :].clone()

#         self.fully_conv2.conv1 = new_layer
#         self.fully_conv2.conv1.weight = torch.nn.Parameter(new_layer.weight)


#         # Combining networks in linear layer
#         self.linear = nn.Linear(num_ftrs1 + num_ftrs2, num_classes)
#         self.sigm = nn.Sigmoid()



#     def forward(self, inputs1, inputs2):
#         output1 = self.fully_conv1(inputs1)
#         output2 = self.fully_conv2(inputs2)
#         concat_output = torch.cat((output1, output2), 1)
#         return self.sigm(self.linear(concat_output))



# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False

# class SingleNetwork(nn.Module):
#     """
#     This class takes one pretrained network,
#     the first conv layer can be modified to take an extra channel.

#     functions: forward
#     """

#     def __init__(self, pretrained_net, weight_init=None):
#         super(SingleNetwork, self).__init__()

#         _, num_classes = get_classes_list()
#         num_ftrs = pretrained_net.fc.in_features

#         if weight_init is not None:
#             layer = pretrained_net.conv1
#             new_layer = nn.Conv2d(in_channels = 4,
#                               out_channels=layer.out_channels,
#                               kernel_size=layer.kernel_size,
#                               stride=layer.stride,
#                               padding=layer.padding,
#                               bias=layer.bias)
#             with torch.no_grad():
#                 new_layer.weight[:,:3,:,:] = layer.weight.clone()


#             if weight_init == "kaiminghe":
#                 torch.nn.init.xavier_uniform_(new_layer.weight[:,3:,:,:])

#             pretrained_net.conv1 = new_layer
#             pretrained_net.conv1.weight = torch.nn.Parameter(new_layer.weight)

        
#         pretrained_net.fc = nn.Linear(num_ftrs, num_classes)

#         self.net = pretrained_net
#         self.sigm = nn.Sigmoid()


#     def forward(self, inputs, **kwargs):
#         return self.sigm(self.net(inputs))
