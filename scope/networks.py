import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset



class ResBlock(nn.Module):
    
    def __init__(self,hidden_dimension,t_size):
        super(ResBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dimension,2*hidden_dimension).double()
        self.relu = nn.LeakyReLU()
        self.adaLN = StyleAdaptiveLayerNorm(2*hidden_dimension,t_size).double()
        self.fc2 = nn.Linear(2*hidden_dimension,hidden_dimension).double() 

    def forward(self,x,y):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.adaLN(out,y)
        out = self.fc2(out)
        return out + x
    




class UNetWithLinear(nn.Module):
    def __init__(self, x_size,t_size,output_size,hidden_size=64,n_layers=4):
        super(UNetWithLinear, self).__init__()
        self.encoder = nn.Linear(x_size, hidden_size).double() 
        self.trans_lst = nn.ModuleList([ResBlock(hidden_size,t_size) for i in range(n_layers)])

        self.decoder =  nn.Linear(hidden_size, output_size).double() 

    def forward(self, x_t, y_t):
        x_t = self.encoder(x_t)
        for trans in self.trans_lst:
            x_t = trans(x_t,y_t)

        x_t = self.decoder(x_t)
        return x_t





class scale_model_muti(nn.Module):
    def __init__(self,output_size,hidden_size=16,input_size=2):
        super(scale_model_muti, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size).double()
        self.fc2 = nn.Linear(hidden_size,output_size).double()
    def forward(self,t,stage):
        x=torch.cat([t,stage],dim=1).double()
        x=F.leaky_relu(self.fc1(x))
        x=self.fc2(x)
        return x



    
class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels

        self.saln = nn.Linear(cond_channels, in_channels * 2)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[:self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels:], 0)

    def forward(self, x, c):
        """
        Parameters:
        x (Tensor): The input feature maps with shape [batch_size, time, in_channels].
        c (Tensor): The conditioning input with shape [batch_size, 1, cond_channels].
        
        Returns:
        Tensor: The modulated feature maps with the same shape as input x.
        """
        saln_params = self.saln(c)
        gamma, beta = torch.chunk(saln_params, chunks=2, dim=-1)
        
        out = self.norm(x)
        out = gamma * out + beta
        
        return out

    
