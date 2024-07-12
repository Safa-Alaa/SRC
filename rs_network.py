
#The code incorporates elements from the following GitHub repositories. We express our gratitude to the authors, acknowledge their contributions, and give them the appropriate credit for their open-source code.
#https://github.com/xudejing/video-clip-order-prediction/blob/master/models/vcopn.py
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class RSNetwork(nn.Module):
    
    def __init__(self, base_network, feature_size, tuple_len, class_num, args):
        """
        Args:
            feature_size (int): 512
        """
        super(RSNetwork, self).__init__()
        self.args = args
        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = class_num 
        
        
        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU(inplace=True)

        self.fully_connected = nn.Sequential(nn.Linear(self.feature_size, 512),
                                  nn.Dropout (0.6),
                                  nn.Linear(512, self.class_num)
                                  )
        #self.fully_connected_head = nn.Sequential(nn.Linear(self.feature_size, 512),
        #                          nn.Dropout (0.6),
        #                          nn.Linear(512, 4)
        #                          )

    def forward(self, tuple):
        
        if self.args.pretext_mode == 'rotation_sorting':
            
            f = []  # clip features
            for i in range(self.tuple_len):
                clip = tuple[:, i, :, :, :, :]
                f.append(self.base_network(clip))

            pf = []  # pairwise concat
            for i in range(self.tuple_len):
                for j in range(i+1, self.tuple_len):
                    pf.append(torch.cat([f[i], f[j]], dim=1))

            pf = [self.fc7(i) for i in pf]
            pf = [self.relu(i) for i in pf]
            h = torch.cat(pf, dim=1)
            h = self.dropout(h)
            h = self.fc8(h)  # logits

            return h
        elif self.args.pretext_mode == 'rotation':
     
            x =  self.base_network(tuple)   # the tuple is just one clip        
            x =  self.fully_connected(x)
         
            return (x)
        
        elif self.args.pretext_mode == 'rs_rotation_multi':
            f = []  # clip features
            for i in range(self.tuple_len):
                clip = tuple[:, i, :, :, :, :]
                f.append(self.base_network(clip))

            pf = []  # pairwise concat
            for i in range(self.tuple_len):
                for j in range(i+1, self.tuple_len):
                    pf.append(torch.cat([f[i], f[j]], dim=1))

            pf = [self.fc7(i) for i in pf]
            pf = [self.relu(i) for i in pf]
            h = torch.cat(pf, dim=1)
            h = self.dropout(h)
            h = self.fc8(h)  # logits
            
            
            
            x_0 =  self.base_network(tuple[:, 0, :, :, :, :])      
            #x_0 =  self.fully_connected_head(x_0)
            
            x_1 =  self.base_network(tuple[:, 1, :, :, :, :])      
            #x_1 =  self.fully_connected_head(x_1)
            
            x_2 =  self.base_network(tuple[:, 2, :, :, :, :])      
            #x_2 =  self.fully_connected_head(x_2)
            
            x_3 =  self.base_network(tuple[:, 3, :, :, :, :])      
            #x_3 =  self.fully_connected_head(x_3)
            
            

            return h, x_0, x_1, x_2, x_3
            
            
            
            

